//! Additional op coverage — at least one test per advertised op.

use super::*;

#[test]
fn arithmetic_is_perfect_square_true() {
    assert_eq_expert(
        "arithmetic",
        "is_perfect_square",
        json!({"n": 49}),
        json!(true),
    );
}

#[test]
fn arithmetic_is_perfect_square_false() {
    assert_eq_expert(
        "arithmetic",
        "is_perfect_square",
        json!({"n": 50}),
        json!(false),
    );
}

#[test]
fn arithmetic_from_base_hex() {
    assert_eq_expert(
        "arithmetic",
        "from_base",
        json!({"s": "ff", "base": 16}),
        json!(255),
    );
}

#[test]
fn arithmetic_from_base_binary() {
    assert_eq_expert(
        "arithmetic",
        "from_base",
        json!({"s": "1010", "base": 2}),
        json!(10),
    );
}

#[test]
fn arithmetic_percent_increase() {
    assert_approx(
        "arithmetic",
        "percent_increase",
        json!({"n": 100, "pct": 20}),
        120.0,
        1e-9,
    );
}

#[test]
fn arithmetic_percent_decrease() {
    assert_approx(
        "arithmetic",
        "percent_decrease",
        json!({"n": 100, "pct": 25}),
        75.0,
        1e-9,
    );
}

#[test]
fn unit_info_km() {
    if let Some(v) = call("unit", "unit_info", json!({"unit": "km"})) {
        assert_eq!(v.get("group").and_then(|g| g.as_str()), Some("length"));
        assert_eq!(v.get("to_si").and_then(|x| x.as_f64()), Some(1000.0));
    }
}

#[test]
fn unit_list_length_group() {
    if let Some(v) = call("unit", "list_units", json!({"group": "length"})) {
        let ids: Vec<&str> = v
            .as_array()
            .expect("array")
            .iter()
            .filter_map(|x| x.as_str())
            .collect();
        assert!(ids.contains(&"km"));
        assert!(ids.contains(&"mi"));
        assert!(
            !ids.contains(&"kg"),
            "length group must not contain mass unit"
        );
    }
}

#[test]
fn unit_list_all() {
    if let Some(v) = call("unit", "list_units", json!({})) {
        let arr = v.as_array().expect("array");
        assert!(arr.len() > 20, "expected >20 units, got {}", arr.len());
    }
}

#[test]
fn statistics_variance() {
    assert_approx(
        "statistics",
        "variance",
        json!({"values": [2,4,4,4,5,5,7,9]}),
        4.0,
        1e-12,
    );
}

#[test]
fn statistics_sum() {
    assert_approx(
        "statistics",
        "sum",
        json!({"values": [1,2,3,4,5]}),
        15.0,
        1e-12,
    );
}

#[test]
fn statistics_range() {
    assert_approx(
        "statistics",
        "range",
        json!({"values": [1,2,3,4,10]}),
        9.0,
        1e-12,
    );
}

#[test]
fn geometry_circle_circumference() {
    assert_approx(
        "geometry",
        "circle_circumference",
        json!({"r": 10}),
        std::f64::consts::TAU * 10.0,
        1e-9,
    );
}

#[test]
fn geometry_circle_diameter() {
    assert_approx("geometry", "circle_diameter", json!({"r": 5}), 10.0, 1e-12);
}

#[test]
fn geometry_sphere_surface_area() {
    assert_approx(
        "geometry",
        "sphere_surface_area",
        json!({"r": 3}),
        4.0 * std::f64::consts::PI * 9.0,
        1e-9,
    );
}

#[test]
fn geometry_cylinder_volume() {
    assert_approx(
        "geometry",
        "cylinder_volume",
        json!({"r": 2, "h": 5}),
        std::f64::consts::PI * 4.0 * 5.0,
        1e-9,
    );
}

#[test]
fn geometry_cone_volume() {
    assert_approx(
        "geometry",
        "cone_volume",
        json!({"r": 3, "h": 4}),
        std::f64::consts::PI * 9.0 * 4.0 / 3.0,
        1e-9,
    );
}

#[test]
fn geometry_cube_volume() {
    assert_approx("geometry", "cube_volume", json!({"s": 3}), 27.0, 1e-12);
}

#[test]
fn geometry_box_volume() {
    assert_approx(
        "geometry",
        "box_volume",
        json!({"l": 2, "w": 3, "h": 4}),
        24.0,
        1e-12,
    );
}

#[test]
fn geometry_square_area() {
    assert_approx("geometry", "square_area", json!({"s": 5}), 25.0, 1e-12);
}

#[test]
fn geometry_square_perimeter() {
    assert_approx("geometry", "square_perimeter", json!({"s": 5}), 20.0, 1e-12);
}

#[test]
fn geometry_rectangle_area() {
    assert_approx(
        "geometry",
        "rectangle_area",
        json!({"l": 4, "w": 5}),
        20.0,
        1e-12,
    );
}

#[test]
fn geometry_triangle_area_heron() {
    // 3-4-5 right triangle has area 6.
    assert_approx(
        "geometry",
        "triangle_area_heron",
        json!({"a": 3, "b": 4, "c": 5}),
        6.0,
        1e-9,
    );
}

#[test]
fn geometry_trapezoid_area() {
    assert_approx(
        "geometry",
        "trapezoid_area",
        json!({"a": 3, "b": 5, "h": 4}),
        16.0,
        1e-12,
    );
}

#[test]
fn geometry_ellipse_area() {
    assert_approx(
        "geometry",
        "ellipse_area",
        json!({"a": 3, "b": 5}),
        std::f64::consts::PI * 15.0,
        1e-9,
    );
}

#[test]
fn trig_acos_one() {
    assert_approx("trig", "acos", json!({"x": 1}), 0.0, 1e-12);
}

#[test]
fn trig_atan_one() {
    assert_approx(
        "trig",
        "atan",
        json!({"x": 1}),
        std::f64::consts::FRAC_PI_4,
        1e-12,
    );
}

#[test]
fn trig_sec_zero() {
    assert_approx("trig", "sec", json!({"x": 0}), 1.0, 1e-12);
}

#[test]
fn trig_csc_pi_half() {
    assert_approx(
        "trig",
        "csc",
        json!({"x": std::f64::consts::FRAC_PI_2}),
        1.0,
        1e-12,
    );
}

#[test]
fn trig_cot_pi_quarter() {
    assert_approx(
        "trig",
        "cot",
        json!({"x": std::f64::consts::FRAC_PI_4}),
        1.0,
        1e-12,
    );
}

#[test]
fn trig_rad_to_deg() {
    assert_approx(
        "trig",
        "rad_to_deg",
        json!({"rad": std::f64::consts::PI}),
        180.0,
        1e-9,
    );
}

#[test]
fn trig_asin_out_of_range() {
    // asin only defined on [-1, 1]; expert returns explicit null.
    assert_eq_expert("trig", "asin", json!({"x": 2}), Value::Null);
}

#[test]
fn string_ops_rot13() {
    assert_eq_expert("string_ops", "rot13", json!({"s": "hello"}), json!("uryyb"));
}

#[test]
fn string_ops_lowercase() {
    assert_eq_expert(
        "string_ops",
        "lowercase",
        json!({"s": "HELLO"}),
        json!("hello"),
    );
}

#[test]
fn string_ops_length() {
    assert_eq_expert("string_ops", "length", json!({"s": "hello"}), json!(5));
}

#[test]
fn string_ops_length_unicode() {
    // `é` is one character but multiple bytes.
    assert_eq_expert("string_ops", "length", json!({"s": "café"}), json!(4));
}

#[test]
fn string_ops_count_char() {
    assert_eq_expert(
        "string_ops",
        "count_char",
        json!({"s": "banana", "ch": "a"}),
        json!(3),
    );
}

#[test]
fn string_ops_count_substring() {
    // `matches()` counts non-overlapping occurrences: "aaaa" contains "aa" twice.
    assert_eq_expert(
        "string_ops",
        "count_substring",
        json!({"s": "aaaa", "needle": "aa"}),
        json!(2),
    );
}

#[test]
fn string_ops_count_words() {
    assert_eq_expert(
        "string_ops",
        "count_words",
        json!({"s": "hello world foo bar"}),
        json!(4),
    );
}

#[test]
fn string_ops_contains_true() {
    assert_eq_expert(
        "string_ops",
        "contains",
        json!({"s": "hello", "needle": "ell"}),
        json!(true),
    );
}

#[test]
fn string_ops_contains_false() {
    assert_eq_expert(
        "string_ops",
        "contains",
        json!({"s": "hello", "needle": "xyz"}),
        json!(false),
    );
}

#[test]
fn string_ops_starts_with() {
    assert_eq_expert(
        "string_ops",
        "starts_with",
        json!({"s": "hello", "prefix": "hel"}),
        json!(true),
    );
}

#[test]
fn string_ops_ends_with() {
    assert_eq_expert(
        "string_ops",
        "ends_with",
        json!({"s": "hello", "suffix": "llo"}),
        json!(true),
    );
}

#[test]
fn hash_hex_decode() {
    assert_eq_expert("hash", "hex_decode", json!({"s": "616263"}), json!("abc"));
}

#[test]
fn hash_hex_decode_with_prefix() {
    assert_eq_expert("hash", "hex_decode", json!({"s": "0x616263"}), json!("abc"));
}

#[test]
fn hash_url_decode() {
    assert_eq_expert(
        "hash",
        "url_decode",
        json!({"s": "hello%20world"}),
        json!("hello world"),
    );
}

#[test]
fn logic_simplify_double_negation() {
    assert_eq_expert(
        "logic",
        "simplify",
        json!({"expr": "NOT NOT A"}),
        json!("A"),
    );
}

#[test]
fn finance_present_value() {
    // PV of 1100 at 10% for 1 year = 1000.
    assert_approx(
        "finance",
        "present_value",
        json!({"fv": 1100, "rate_pct": 10, "years": 1}),
        1000.0,
        1e-9,
    );
}

#[test]
fn finance_simple_interest() {
    assert_approx(
        "finance",
        "simple_interest",
        json!({"principal": 1000, "rate_pct": 5, "years": 3}),
        150.0,
        1e-9,
    );
}

#[test]
fn finance_mortgage_payment() {
    // 100k at 6% over 30 years ≈ $599.55/mo.
    assert_approx(
        "finance",
        "mortgage_payment",
        json!({"principal": 100000, "annual_rate_pct": 6, "years": 30}),
        599.55,
        1.0,
    );
}

#[test]
fn finance_bayes() {
    // P(B|A)=0.9, P(A)=0.01, P(B)=0.1 → P(A|B) = 0.09.
    assert_approx(
        "finance",
        "bayes",
        json!({"p_b_given_a": 0.9, "p_a": 0.01, "p_b": 0.1}),
        0.09,
        1e-9,
    );
}

#[test]
fn finance_bayes_p_b_zero() {
    assert_eq_expert(
        "finance",
        "bayes",
        json!({"p_b_given_a": 0.9, "p_a": 0.1, "p_b": 0}),
        Value::Null,
    );
}

#[test]
fn element_by_symbol() {
    assert_field(
        "element",
        "by_symbol",
        json!({"symbol": "Au"}),
        "name",
        json!("gold"),
    );
}

#[test]
fn element_by_symbol_case_insensitive() {
    assert_field(
        "element",
        "by_symbol",
        json!({"symbol": "fe"}),
        "name",
        json!("iron"),
    );
}

#[test]
fn element_list() {
    if let Some(v) = call("element", "list", json!({})) {
        let arr = v.as_array().expect("array");
        assert_eq!(arr.len(), 118, "expected 118 elements");
    }
}

#[test]
fn isbn_isbn10_to_isbn13() {
    assert_eq_expert(
        "isbn",
        "isbn10_to_isbn13",
        json!({"isbn": "0-306-40615-2"}),
        json!("9780306406157"),
    );
}

#[test]
fn isbn_isbn13_to_isbn10() {
    assert_eq_expert(
        "isbn",
        "isbn13_to_isbn10",
        json!({"isbn": "978-0-596-52068-7"}),
        json!("0596520689"),
    );
}

#[test]
fn conway_step_blinker() {
    // A horizontal blinker → vertical blinker after one step.
    if let Some(v) = call("conway", "step", json!({"grid": [[0,0,0],[1,1,1],[0,0,0]]})) {
        assert_eq!(v, json!([[0, 1, 0], [0, 1, 0], [0, 1, 0]]));
    }
}

#[test]
fn graph_topological_sort_dag() {
    if let Some(v) = call(
        "graph",
        "topological_sort",
        json!({
            "edges": [["A","B"],["B","C"],["A","C"]],
            "directed": true
        }),
    ) {
        let order: Vec<&str> = v
            .as_array()
            .expect("array")
            .iter()
            .filter_map(|x| x.as_str())
            .collect();
        // Any valid topo order places A before B and B before C.
        let ai = order.iter().position(|&n| n == "A").expect("A present");
        let bi = order.iter().position(|&n| n == "B").expect("B present");
        let ci = order.iter().position(|&n| n == "C").expect("C present");
        assert!(ai < bi && bi < ci, "invalid topo order: {:?}", order);
    }
}

#[test]
fn graph_topological_sort_cycle_returns_null() {
    assert_eq_expert(
        "graph",
        "topological_sort",
        json!({"edges": [["A","B"],["B","C"],["C","A"]], "directed": true}),
        Value::Null,
    );
}
