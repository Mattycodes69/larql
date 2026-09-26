//! StreamingWeights

use super::*;

mod streaming {
    use super::*;
    use safetensors::tensor::TensorView;
    use safetensors::Dtype;

    const Q_KEY: &str = "model.layers.0.self_attn.q_proj.weight";
    const NORM_KEY: &str = "model.norm.weight";
    const OUTPUT_KEY: &str = "output.weight";
    const BF16_KEY: &str = "experts.packed";
    const Q_ROWS: usize = 2;
    const Q_COLS: usize = 3;

    fn f32_bytes(data: &[f32]) -> Vec<u8> {
        data.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// One serialized safetensors shard with a 2D f32, 1D f32, lm_head
    /// fallback, and a BF16 packed tensor.
    fn shard() -> (Vec<u8>, HashMap<String, (usize, String)>) {
        let q = f32_bytes(&(0..Q_ROWS * Q_COLS).map(|i| i as f32).collect::<Vec<_>>());
        let norm = f32_bytes(&[1.0, 2.0, 3.0]);
        let output = f32_bytes(&[4.0, 5.0, 6.0, 7.0]);
        let bf16 = vec![0x80u8, 0x3f, 0x00, 0x40, 0x40, 0x40, 0x80, 0x40];
        let views = vec![
            (
                Q_KEY,
                TensorView::new(Dtype::F32, vec![Q_ROWS, Q_COLS], &q).unwrap(),
            ),
            (
                NORM_KEY,
                TensorView::new(Dtype::F32, vec![3], &norm).unwrap(),
            ),
            (
                OUTPUT_KEY,
                TensorView::new(Dtype::F32, vec![2, 2], &output).unwrap(),
            ),
            (
                BF16_KEY,
                TensorView::new(Dtype::BF16, vec![2, 2], &bf16).unwrap(),
            ),
        ];
        let blob = safetensors::serialize(views, None).unwrap();
        let index = [Q_KEY, NORM_KEY, OUTPUT_KEY, BF16_KEY]
            .into_iter()
            .map(|k| (k.to_string(), (0usize, k.to_string())))
            .collect();
        (blob, index)
    }

    #[test]
    fn streaming_source_reads_tensors_vectors_and_lm_head() {
        let (blob, index) = shard();
        let arch = larql_models::detect_from_json(&qwen2_arch_json());
        let shards: Vec<&[u8]> = vec![&blob];
        let source = StreamingWeights {
            shard_mmaps: &shards,
            tensor_index: &index,
            arch: &*arch,
            num_layers: 1,
        };

        let (data, rows, cols) = source.get_tensor(Q_KEY).expect("2D tensor");
        assert_eq!((rows, cols), (Q_ROWS, Q_COLS));
        assert_eq!(
            data,
            (0..Q_ROWS * Q_COLS).map(|i| i as f32).collect::<Vec<_>>()
        );
        // Rank mismatches return None rather than mis-shaping.
        assert!(source.get_tensor(NORM_KEY).is_none());
        assert!(source.get_vector(Q_KEY).is_none());
        assert_eq!(source.get_vector(NORM_KEY), Some(vec![1.0, 2.0, 3.0]));
        assert!(source.get_tensor("nope").is_none());

        // lm_head falls back from lm_head.weight to output.weight.
        let (head, r, c) = source.lm_head().expect("lm_head fallback");
        assert_eq!((r, c), (2, 2));
        assert_eq!(head, vec![4.0, 5.0, 6.0, 7.0]);

        // vector_names picks norm/bias-shaped keys only, sorted.
        assert_eq!(source.vector_names(), vec![NORM_KEY.to_string()]);
        assert_eq!(source.num_layers(), 1);
        assert_eq!(source.arch().config().hidden_size, HIDDEN);
    }

    #[test]
    fn streaming_decodes_f16_bf16_and_rejects_unsupported_dtypes() {
        // f16 1.0 = 0x3C00, 2.0 = 0x4000; bf16 1.0 = 0x3F80, 2.0 = 0x4000.
        let f16 = vec![0x00u8, 0x3C, 0x00, 0x40];
        let bf16 = vec![0x80u8, 0x3F, 0x00, 0x40];
        let ints = vec![0u8; 8];
        let views = vec![
            (
                "half",
                TensorView::new(Dtype::F16, vec![1, 2], &f16).unwrap(),
            ),
            (
                "brain",
                TensorView::new(Dtype::BF16, vec![1, 2], &bf16).unwrap(),
            ),
            (
                "ints",
                TensorView::new(Dtype::I32, vec![1, 2], &ints).unwrap(),
            ),
        ];
        let blob = safetensors::serialize(views, None).unwrap();
        let index: HashMap<String, (usize, String)> = ["half", "brain", "ints"]
            .into_iter()
            .map(|k| (k.to_string(), (0usize, k.to_string())))
            .collect();
        let arch = larql_models::detect_from_json(&qwen2_arch_json());
        let shards: Vec<&[u8]> = vec![&blob];
        let source = StreamingWeights {
            shard_mmaps: &shards,
            tensor_index: &index,
            arch: &*arch,
            num_layers: 1,
        };

        assert_eq!(source.get_tensor("half"), Some((vec![1.0, 2.0], 1, 2)));
        assert_eq!(source.get_tensor("brain"), Some((vec![1.0, 2.0], 1, 2)));
        assert!(source.get_tensor("ints").is_none(), "I32 is unsupported");
        // No lm_head.weight / output.weight anywhere → None.
        assert!(source.lm_head().is_none());
    }

    #[test]
    fn streaming_packed_bf16_requires_bf16_dtype() {
        let (blob, index) = shard();
        let arch = larql_models::detect_from_json(&qwen2_arch_json());
        let shards: Vec<&[u8]> = vec![&blob];
        let source = StreamingWeights {
            shard_mmaps: &shards,
            tensor_index: &index,
            arch: &*arch,
            num_layers: 1,
        };
        let raw = source.get_packed_bf16(BF16_KEY).expect("bf16 bytes");
        assert_eq!(raw.len(), 2 * 2 * 2);
        assert!(
            source.get_packed_bf16(Q_KEY).is_none(),
            "f32 is not packed bf16"
        );
        assert!(source.get_packed_bf16("missing").is_none());
    }

    /// `get_raw_u8` is what lets the per-expert writer reach a packed
    /// MXFP4 layer's `*_blocks`/`*_scales` on the streaming path — without
    /// it the synthesised per-expert keys miss and the expert store is
    /// silently absent (the 2026-08-09 gpt-oss-20b extraction).
    #[test]
    fn streaming_raw_u8_returns_bytes_and_shape_for_u8_only() {
        let u8_data = vec![7u8; 12];
        let f32_data: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let views = vec![
            (
                "blocks",
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::U8,
                    vec![1, 3, 4],
                    &u8_data,
                )
                .unwrap(),
            ),
            (
                "floats",
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    vec![1, 2],
                    &f32_data,
                )
                .unwrap(),
            ),
        ];
        let blob = safetensors::serialize(views, None).unwrap();
        let index: HashMap<String, (usize, String)> = ["blocks", "floats"]
            .into_iter()
            .map(|k| (k.to_string(), (0usize, k.to_string())))
            .collect();
        let arch = larql_models::detect_from_json(&qwen2_arch_json());
        let shards: Vec<&[u8]> = vec![&blob];
        let source = StreamingWeights {
            shard_mmaps: &shards,
            tensor_index: &index,
            arch: &*arch,
            num_layers: 1,
        };

        let (bytes, shape) = source.get_raw_u8("blocks").expect("u8 tensor");
        assert_eq!(bytes, u8_data);
        assert_eq!(shape, vec![1, 3, 4], "shape must survive for layout checks");
        assert!(
            source.get_raw_u8("floats").is_none(),
            "non-U8 dtypes are not raw quantised blocks"
        );
        assert!(source.get_raw_u8("missing").is_none());
    }
}
