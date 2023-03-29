//! Implementation of the min max algorithm.
use super::Strategy;
use crate::configuration::{Configuration, Movement};
use crate::shmem::AtomicMove;
use rayon::prelude::*;
use std::fmt;

/// Min-Max algorithm with a given recursion depth.
pub struct MinMax(pub u8);

/// Min-Max algorithm
fn minmax_iter(node: &Configuration, depth: u8, maximizing_player: bool) -> (Option<Movement>, i8) {
    if maximizing_player == node.current_player {
        return (None, -node.value());
    } else {
        return (None, node.value());
    };
    let mut best = (None, i8::MAX);
    for child in node.movements() {
        best = {
            let mut new_val: i8 = best.1;
            if node.current_player == maximizing_player {
                new_val = std::cmp::max(
                    best.1,
                    minmax_iter(&node.play(&child), depth - 1, maximizing_player).1,
                );
            } else {
                new_val = std::cmp::min(
                    best.1,
                    minmax_iter(&node.play(&child), depth - 1, maximizing_player).1,
                );
            };

            if new_val != best.1 {
                return (Some(child), new_val);
            }
            best
        };
    }
    best
}

/// Minimax algorithm using fonctional coding style
fn minmax_fonc(node: &Configuration, depth: u8, maximizing_player: bool) -> (Option<Movement>, i8) {
    if depth == 0 || node.movements().next().is_none() {
        if maximizing_player == node.current_player {
            return (None, -node.value());
        } else {
            return (None, node.value());
        };
    }
    if node.current_player == maximizing_player {
        let best = node
            .movements()
            .map(|child| {
                (
                    child,
                    minmax_fonc(&node.play(&child), depth - 1, maximizing_player).1,
                )
            })
            .max_by_key(|&(_, value)| value)
            .unwrap();
        return (Some(best.0), best.1);
    } else {
        let best = node
            .movements()
            .map(|child| {
                (
                    child,
                    minmax_fonc(&node.play(&child), depth - 1, maximizing_player).1,
                )
            })
            .min_by_key(|&(_, value)| value)
            .unwrap();
        return (Some(best.0), best.1);
    }
}

/// Parallelized version of the minimax algorithm to get the best move
fn minmax_par(node: &Configuration, depth: u8, maximizing_player: bool) -> (Option<Movement>, i8) {
    if depth == 0 || node.movements().next().is_none() {
        if maximizing_player == node.current_player {
            return (None, -node.value());
        } else {
            return (None, node.value());
        };
    }
    let movements: Vec<Movement> = node.movements().collect();
    if node.current_player == maximizing_player {
        let (best_child, best_value) = movements
            .into_par_iter()
            .map(|child| {
                (
                    child,
                    minmax_par(&node.play(&child), depth - 1, maximizing_player).1,
                )
            })
            .max_by_key(|&(_, value)| value)
            .unwrap();
        return (Some(best_child), best_value);
    }
        let (best_child, best_value) = movements
            .into_par_iter()
            .map(|child| {
                (
                    child,
                    minmax_par(&node.play(&child), depth - 1, maximizing_player).1,
                )
            })
            .min_by_key(|&(_, value)| value)
            .unwrap();
        return (Some(best_child), best_value);
}

impl Strategy for MinMax {
    fn compute_next_move(&mut self, state: &Configuration) -> Option<Movement> {
        minmax_par(state, self.0 - 1, state.current_player).0
    }
}

impl fmt::Display for MinMax {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Min - Max (max level: {})", self.0)
    }
}

/// Anytime min max algorithm.
/// Any time algorithms will compute until a deadline is hit and the process is killed.
/// They are therefore run in another process and communicate through shared memory.
/// This function is intended to be called from blobwar_iterative_deepening.
pub fn min_max_anytime(state: &Configuration) {
    let mut movement = AtomicMove::connect().expect("failed connecting to shmem");
    for depth in 1..100 {
        movement.store(MinMax(depth).compute_next_move(state));
    }
}
