//! Every resident form the CPU loader produces, against its declaration
//! A prepared plan reconciles, and the census is a third reading
//! Providers: gone or changed invalidates the image
//! The four-way correspondence

use super::*;

/// For each resident form: load the object, price the pin, reconcile.
/// The pricing that comes from the executor's block geometry is then
/// re-priced under a MUTATED geometry, and the reconciliation must break —
/// otherwise the comparison would be reading the declaration twice.
#[test]
#[serial_test::serial]
fn every_resident_form_reconciles_with_its_declaration_and_a_mutated_geometry_breaks_it() {
    let f = fixture(dense_f32_model, None);
    let op = an_ffn_projection(&f.plan);
    let stored = |o: &OperandRef| f.store.stored_len(o);
    let executor = BlockGeometry::executor();
    let mutated = BlockGeometry {
        q8_block: executor.q8_block / 2,
        q4_block: executor.q4_block / 2,
        q8_indexed: !executor.q8_indexed,
    };
    let cases: [(WeightFormat, RealizationForm, bool); 6] = [
        (
            WeightFormat::F32,
            RealizationForm::Decode(PhysicalProjectionPlan::BlasF32),
            false,
        ),
        (
            WeightFormat::Q8,
            RealizationForm::Requantise(PhysicalProjectionPlan::FusedQ8),
            true,
        ),
        (
            WeightFormat::Q4,
            RealizationForm::Requantise(PhysicalProjectionPlan::FusedQ4),
            true,
        ),
        (
            WeightFormat::F16,
            RealizationForm::DeviceResident(WeightFormat::F16),
            false,
        ),
        (
            WeightFormat::Nvfp4,
            RealizationForm::DeviceResident(WeightFormat::Nvfp4),
            false,
        ),
        (
            WeightFormat::Mxfp4,
            RealizationForm::DeviceResident(WeightFormat::Mxfp4),
            false,
        ),
    ];
    for (format, form, geometry_priced) in cases {
        let loaded = load_weight((&f.store).into(), &op, format)
            .unwrap_or_else(|e| panic!("{format:?}: {e}"));
        let rec = record(&op, RealizationId::cpu(form), ResidencyProfile::DECODED_F32);
        let observed = vec![Bound::one(&op, &loaded)
            .observed(rec.planned.operation, rec.planned.layer)
            .unwrap()];
        let expected = expectations(std::slice::from_ref(&rec), stored, executor);
        let ok = reconcile(&expected, &observed).unwrap_or_else(|e| panic!("{format:?}: {e}"));
        assert_eq!(ok.matched, 1);
        assert!(
            ok.padding < loaded.padded_allocations() as u64 * DEVICE_PAGE_ALIGN as u64 + 1,
            "{format:?}: padding {} over {} padded allocation(s)",
            ok.padding,
            loaded.padded_allocations()
        );
        if loaded.padded_allocations() == 0 {
            assert_eq!(ok.padding, 0, "{format:?}: an exact form has no padding");
        }
        let under_mutation = expectations(std::slice::from_ref(&rec), stored, mutated);
        let broken = reconcile(&under_mutation, &observed).is_err();
        assert_eq!(
            broken, geometry_priced,
            "{format:?}: a mutated block geometry must break exactly the forms it prices"
        );
    }

    // ── And one real Direct case, on a container that can hold one ────
    //
    // Every form above is either priced from the executor's geometry or
    // decoded to f32, so none of them reaches the branch that reads
    // `selection.residency` for a pin that binds the STORED bytes in
    // place. That branch is the entire reason residency had to become
    // realization-scoped, so it is witnessed here at the same
    // declaration-versus-object boundary: a genuinely compiled Q6_K
    // pack, the pin taken from the production selector rather than
    // written down, and the object the production loader bound.
    let tmp = tempfile::tempdir().unwrap();
    let (_src, pack) = compiled(&tmp, Q6_K);
    let store = open_pack(&pack, Q6_K);
    let packed = a_stored_matrix(&pack, Q6_K);
    let packed_stored = |o: &OperandRef| store.stored_len(o);
    let operation = Operation::Project(MatrixClass::FfnProjection);
    let planned = PlannedOperand {
        operand: packed.clone(),
        operation,
        access: operation.access(),
        extent: RepresentationExtent::BASE,
        layer: Some(0),
        declared_representation: None,
        logical_elements: packed.shape.iter().product(),
    };
    let facts = RepresentationFacts::resolve(&packed.dtype);

    // SELECTION: one stored operand, two production answers.
    let direct = select_cpu(&planned, &facts, KQuantExecution::Direct).unwrap();
    let widened = select_cpu(&planned, &facts, KQuantExecution::Widen).unwrap();
    assert_eq!(
        direct.realization,
        RealizationId::cpu(RealizationForm::Direct(PhysicalProjectionPlan::FusedKQuant)),
        "the production selector pins the stored pack in place"
    );
    assert_eq!(
        widened.realization,
        RealizationId::cpu(RealizationForm::Decode(PhysicalProjectionPlan::BlasF32)),
        "and widening the same operand decodes it"
    );
    // DECLARATION: the two pins do not describe the same residency.
    assert!(
        direct.residency.bytes_per_weight < widened.residency.bytes_per_weight,
        "the pins declare {} and {} bytes per weight over identical stored bytes",
        direct.residency.bytes_per_weight,
        widened.residency.bytes_per_weight
    );

    let pinned = |selection: &Selection| RealizationRecord {
        planned: planned.clone(),
        representation: packed.dtype.clone(),
        codec_provider: None,
        lowering_provider: LoweringIdentity::cpu_production(),
        extent: ExtentPin::unknown(),
        verified_bytes: 0,
        dependencies: Vec::new(),
        selection: selection.clone(),
    };
    let bind = |format| {
        let loaded = load_weight((&store).into(), &packed, format)
            .unwrap_or_else(|e| panic!("{format:?}: {e}"));
        let observed = vec![Bound::one(&packed, &loaded)
            .observed(operation, Some(0))
            .unwrap()];
        observed
    };
    let on_stored = bind(WeightFormat::KQuant);
    let on_widened = bind(WeightFormat::F32);

    // OBSERVATION: the genuine Direct declaration against the object the
    // loader actually bound — the stored blocks, not a derivative.
    let direct_rec = pinned(&direct);
    let priced = expectations(std::slice::from_ref(&direct_rec), packed_stored, executor);
    let ok = reconcile(&priced, &on_stored).expect("the Direct pin reconciles with what it bound");
    assert_eq!(ok.matched, 1);
    assert_eq!(ok.padding, 0, "stored blocks are bound exactly");
    // A Direct pin is priced from its OWN declaration, so the executor's
    // block geometry is not what holds this arm up.
    reconcile(
        &expectations(std::slice::from_ref(&direct_rec), packed_stored, mutated),
        &on_stored,
    )
    .expect("a Direct pin is priced from its declaration, not the executor's geometry");

    // MUTATION: move the Direct pin's residency and nothing else. The
    // reconciliation must fail, and name the pin it failed for.
    let mut tampered = direct_rec.clone();
    tampered.selection.residency.bytes_per_weight *= 2.0;
    let err = reconcile(
        &expectations(std::slice::from_ref(&tampered), packed_stored, executor),
        &on_stored,
    )
    .expect_err("a Direct declaration that doubled must not reconcile with the bound object")
    .to_string();
    assert!(err.contains(&packed.tensor), "{err}");
    assert!(err.contains("cpu:direct/FusedKQuant"), "{err}");
    assert!(
        err.contains("the declaration and the loader disagree"),
        "{err}"
    );

    // CONTROL: the decode pin on the same stored bytes is undisturbed by
    // that mutation — it was a fact about one pin, not about the operand.
    let decode_rec = pinned(&widened);
    let ok = reconcile(
        &expectations(std::slice::from_ref(&decode_rec), packed_stored, executor),
        &on_widened,
    )
    .expect("the decode pin still reconciles");
    assert_eq!(ok.matched, 1);
    reconcile(
        &expectations(std::slice::from_ref(&direct_rec), packed_stored, executor),
        &on_stored,
    )
    .expect("and so does the un-tampered Direct pin");
}

#[test]
fn the_q8_pricing_follows_the_executor_s_index_flag_and_block() {
    let plain = BlockGeometry {
        q8_block: 64,
        q4_block: 64,
        q8_indexed: false,
    };
    let indexed = BlockGeometry {
        q8_indexed: true,
        ..plain
    };
    let p = resident_profile_with(WeightFormat::Q8, plain).bytes_per_weight;
    let i = resident_profile_with(WeightFormat::Q8, indexed).bytes_per_weight;
    assert!((p - (1.0 + 4.0 / 64.0)).abs() < 1e-12);
    assert!(
        (i - (1.0 + 6.0 / 64.0)).abs() < 1e-12,
        "one i16 sum per block"
    );
    let q4 = resident_profile_with(WeightFormat::Q4, plain).bytes_per_weight;
    assert!((q4 - (0.5 + 4.0 / 64.0)).abs() < 1e-12);
}

#[test]
fn a_prepared_dense_plan_reconciles_every_pin_and_the_census_is_the_same_bytes() {
    let f = fixture(dense_f32_model, None);
    let ops = prepared(&f);
    let done = ops.reconcile(&f.plan, (&f.store).into()).unwrap();
    assert_eq!(done.matched, ops.realizations().len());
    assert_eq!(done.padding, 0, "f32 images are exact");
    // The census walks the loaded objects by site; the observation walks
    // them by operand. Two readings of one resident image must agree.
    let observed = ops.bound(&f.plan).unwrap();
    let total: u64 = observed.iter().map(|o| o.resident_bytes).sum();
    let census = ops.residency_census();
    assert_eq!(total, (census.total() - census.glue.total()) as u64);
    assert_eq!(done.observed_resident, total);
}

#[test]
fn the_entropy_coded_container_s_stored_bytes_are_the_recorded_length_and_its_staging_is_the_image()
{
    let f = fixture(dense_f32_model, Some(Transcode::Bf16Zlib));
    let ops = prepared(&f);
    let expected = ops.expectations((&f.store).into(), BlockGeometry::executor());
    let mut seen = 0;
    for e in expected
        .iter()
        .filter(|e| f.store.stored_dtype(&e.operand) == Some("BF16_ZLIB"))
    {
        let recorded = f.store.stored_len(&e.operand).unwrap();
        assert_eq!(e.stored_bytes, recorded, "{}", e.operand.tensor);
        assert_ne!(
            e.stored_bytes,
            e.logical_elements as u64 * BF16_WIDTH,
            "instance, not shape"
        );
        assert_eq!(
            e.staging,
            e.logical_elements as u64 * F32_WIDTH,
            "decoded through f32"
        );
        assert_eq!(e.declared_resident, e.logical_elements as u64 * F32_WIDTH);
        assert_eq!(e.working_set(), e.staging + e.declared_resident);
        seen += 1;
    }
    assert!(seen > 0);
    ops.reconcile(&f.plan, (&f.store).into()).unwrap();
    // Footprint and touch: every operand is read once per operation, and
    // no operand is shared, so the two agree on this plan.
    assert_eq!(
        stored_footprint(&expected).bytes,
        execution_touch(&expected)
    );
}

#[test]
fn a_tied_head_counts_once_in_the_stored_footprint_and_once_per_operation_in_the_touch() {
    let f = tied_head();
    let ops = prepared(&f);
    let expected = ops.expectations((&f.store).into(), BlockGeometry::executor());
    let table = f.plan.embedding.as_ref().unwrap().table.clone();
    let table_bytes = f.store.stored_len(&table).unwrap();
    let footprint = stored_footprint(&expected);
    let touch = execution_touch(&expected);
    let operations = expected.len();
    assert_eq!(
        footprint.operands,
        operations - 1,
        "the table is one stored operand under two operations"
    );
    assert_eq!(
        touch,
        footprint.bytes + table_bytes,
        "the table is read once per operation"
    );
    ops.reconcile(&f.plan, (&f.store).into()).unwrap();
}

#[test]
fn a_provider_that_disappears_invalidates_the_preparation_rather_than_falling_back() {
    let f = fixture(dense_f32_model, None);
    // The store carries the registry it decodes through, so selection,
    // provider identity and decode all read one registry.
    let with: &'static CodecRegistry = Box::leak(Box::new(rung_one_registry(true)));
    let ops = PreparedOperands::load(
        &f.plan,
        &f.store_through(with),
        &ProductionBackend::new(),
        ExecutionSlice::Full,
    )
    .unwrap();
    ops.ensure_providers_in(with).unwrap();
    ops.ensure_providers_in(CodecRegistry::builtin())
        .expect("the built-in registry offers the same F32 identity");
    let without: &'static CodecRegistry = Box::leak(Box::new(rung_one_registry(false)));
    let err = ops.ensure_providers_in(without).unwrap_err().to_string();
    assert!(
        err.contains("`F32`") && err.contains("no registered codec"),
        "{err}"
    );
    assert!(err.contains("re-prepare"), "{err}");
    // And preparing through a store whose registry lacks F32 refuses
    // outright.
    let Err(err) = PreparedOperands::load(
        &f.plan,
        &f.store_through(without),
        &ProductionBackend::new(),
        ExecutionSlice::Full,
    ) else {
        panic!("F32 is not registered there");
    };
    let err = err.to_string();
    assert!(err.contains("unregistered representation"), "{err}");
}

/// Planned operands, pins, bound objects and the ledger's account of what
/// ran describe one execution — EXACTLY, in positions. The projection
/// ledger is process-global, so the exact reading is only available in a
/// process that executes nothing else: this test re-launches the test
/// binary on itself, marked, and the marked run makes the assertions.
#[test]
fn planned_operands_pins_bound_objects_and_the_ledger_correspond() {
    const WITNESS: &str = "LARQL_LEDGER_WITNESS";
    const NAME: &str = "format::vindex3::opplan::exec::tests::accounting::planned_operands_pins_bound_objects_and_the_ledger_correspond";
    if std::env::var_os(WITNESS).is_some() {
        return exact_ledger_correspondence();
    }
    let exe = std::env::current_exe().expect("the test binary knows its own path");
    let output = std::process::Command::new(exe)
        .args(["--exact", NAME, "--test-threads=1", "--nocapture"])
        .env(WITNESS, "1")
        .output()
        .expect("the test binary re-launches");
    assert!(
        output.status.success(),
        "the exact correspondence did not hold in an isolated process:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
