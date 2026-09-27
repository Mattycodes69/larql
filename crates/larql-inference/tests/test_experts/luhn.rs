//! luhn
//! markov
//! conway
//! dijkstra
//! graph
//! sql

use super::*;

#[test]
fn luhn_visa_valid() {
    assert_eq_expert(
        "luhn",
        "check",
        json!({"number": "4532015112830366"}),
        json!(true),
    );
}

#[test]
fn luhn_amex_valid() {
    assert_eq_expert(
        "luhn",
        "check",
        json!({"number": "378282246310005"}),
        json!(true),
    );
}

#[test]
fn luhn_invalid() {
    assert_eq_expert(
        "luhn",
        "check",
        json!({"number": "1234567890123456"}),
        json!(false),
    );
}

#[test]
fn luhn_check_digit() {
    assert_eq_expert(
        "luhn",
        "generate_check_digit",
        json!({"number": "453201511283036"}),
        json!(6),
    );
}

#[test]
fn luhn_card_type_amex() {
    assert_eq_expert(
        "luhn",
        "card_type",
        json!({"number": "378282246310005"}),
        json!("amex"),
    );
}

#[test]
fn markov_expected_value() {
    assert_approx(
        "markov",
        "expected_value",
        json!({"outcomes": [1, 2, 3], "probabilities": [0.2, 0.5, 0.3]}),
        2.1,
        1e-9,
    );
}

#[test]
fn markov_steady_state() {
    // Symmetric-ish test: equal columns of the transpose fixed point.
    if let Some(v) = call(
        "markov",
        "steady_state",
        json!({"matrix": [[0.5, 0.5], [0.3, 0.7]]}),
    ) {
        let arr = v.as_array().expect("array");
        let sum: f64 = arr.iter().filter_map(|x| x.as_f64()).sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "probabilities must sum to 1, got {sum}"
        );
    }
}

#[test]
fn conway_blinker_one_gen() {
    if let Some(v) = call(
        "conway",
        "simulate",
        json!({
            "grid": [[0,0,0],[1,1,1],[0,0,0]],
            "generations": 1
        }),
    ) {
        assert_eq!(v.get("live").and_then(|x| x.as_i64()), Some(3));
    }
}

#[test]
fn conway_still_block() {
    // A 2×2 block is a still life — stays at 4 live cells.
    if let Some(v) = call(
        "conway",
        "simulate",
        json!({
            "grid": [[1,1],[1,1]],
            "generations": 1
        }),
    ) {
        assert_eq!(v.get("live").and_then(|x| x.as_i64()), Some(4));
    }
}

#[test]
fn dijkstra_shortest_path() {
    assert_field(
        "dijkstra",
        "shortest_path",
        json!({"edges": [["A","C",2],["C","B",1],["A","B",5]], "from": "A", "to": "B"}),
        "distance",
        json!(3),
    );
}

#[test]
fn dijkstra_reachable() {
    assert_field(
        "dijkstra",
        "reachable",
        json!({"edges": [["A","B"],["B","C"]], "from": "A", "to": "C"}),
        "reachable",
        json!(true),
    );
}

#[test]
fn dijkstra_mst() {
    assert_field(
        "dijkstra",
        "mst",
        json!({"edges": [["A","B",4],["B","C",2],["A","C",5]]}),
        "weight",
        json!(6),
    );
}

#[test]
fn graph_most_central() {
    assert_field(
        "graph",
        "most_central",
        json!({"edges": [["A","B"],["B","C"],["B","D"],["B","E"]]}),
        "node",
        json!("B"),
    );
}

#[test]
fn graph_cycle_detected() {
    assert_eq_expert(
        "graph",
        "has_cycle",
        json!({"edges": [["A","B"],["B","C"],["C","A"]]}),
        json!(true),
    );
}

#[test]
fn graph_connected_components() {
    assert_eq_expert(
        "graph",
        "connected_components",
        json!({"edges": [["A","B"],["C","D"]]}),
        json!(2),
    );
}

#[test]
fn graph_bipartite_yes() {
    assert_eq_expert(
        "graph",
        "is_bipartite",
        json!({"edges": [["A","B"],["B","C"],["C","D"]]}),
        json!(true),
    );
}

#[test]
fn sql_count() {
    assert_eq_expert(
        "sql",
        "execute",
        json!({"sql": "CREATE TABLE t (x int); INSERT INTO t VALUES (1); INSERT INTO t VALUES (2); SELECT COUNT(*) FROM t"}),
        json!(2),
    );
}

#[test]
fn sql_sum() {
    assert_eq_expert(
        "sql",
        "execute",
        json!({"sql": "CREATE TABLE s (v int); INSERT INTO s VALUES (10); INSERT INTO s VALUES (20); INSERT INTO s VALUES (30); SELECT SUM(v) FROM s"}),
        json!(60),
    );
}

#[test]
fn sql_select_with_where() {
    assert_eq_expert(
        "sql",
        "execute",
        json!({"sql": "CREATE TABLE u (id int, name text); INSERT INTO u VALUES (1, 'Alice'); INSERT INTO u VALUES (2, 'Bob'); SELECT name FROM u WHERE id = 2"}),
        json!("Bob"),
    );
}

#[test]
fn sql_avg() {
    assert_eq_expert(
        "sql",
        "execute",
        json!({"sql": "CREATE TABLE a (n int); INSERT INTO a VALUES (10); INSERT INTO a VALUES (20); SELECT AVG(n) FROM a"}),
        json!(15),
    );
}
