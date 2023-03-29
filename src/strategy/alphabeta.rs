//! Alpha - Beta algorithm.
use std::fmt;

use super::Strategy;
use crate::configuration::{Configuration, Movement};
use crate::shmem::AtomicMove;
use rayon::prelude::*;

use std::sync::Mutex;


/// Alpha - Beta algorithm with given maximum number of recursions.
pub struct AlphaBeta(pub u8);

fn alpha_beta_par(
    node: &Configuration,
    depth: u8,
    alpha: i8,
    beta: i8,
    maximizing_player: bool,
) -> (Option<Movement>, i8) {
    if depth == 0 || node.movements().next().is_none() {
        if maximizing_player == node.current_player {
            return (None, -node.value());
        } else {
            return (None, node.value());
        };
    }

    let movements: Vec<Movement> = node.movements().collect();

    if maximizing_player == node.current_player {
        movements
            .into_par_iter()
            .map_init(|| (i8::MIN, i8::MAX), |(local_alpha, _), child| {
                let (_, value) = alpha_beta_par(&node.play(&child), depth - 1, *local_alpha, beta, maximizing_player);
                if value > *local_alpha {
                    *local_alpha = value;
                }
                (Some(child.clone()), value)
            })
            .reduce_with(|(child1, value1), (child2, value2)| {
                if value1 > value2 {
                    (child1, value1)
                } else {
                    (child2, value2)
                }
            })
            .unwrap_or((None, i8::MIN))
    } else {
        movements
            .into_par_iter()
            .map_init(|| (i8::MIN, i8::MAX), |(_, local_beta), child| {
                let (_, value) = alpha_beta_par(&node.play(&child), depth - 1, alpha, *local_beta, maximizing_player);
                if value < *local_beta {
                    *local_beta = value;
                }
                (Some(child.clone()), value)
            })
            .reduce_with(|(child1, value1), (child2, value2)| {
                if value1 < value2 {
                    (child1, value1)
                } else {
                    (child2, value2)
                }
            })
            .unwrap_or((None, i8::MAX))
    }
}

fn alpha_beta(
    node: &Configuration,
    depth: u8,
    mut alpha: i8,
    mut beta: i8,
    maximizing_player: bool,
) -> (Option<Movement>, i8) {
    if depth == 0 || node.movements().next().is_none() {
        if maximizing_player == node.current_player {
            return (None, -node.value());
        } else {
            return (None, node.value());
        };
    }

    let movements: Vec<Movement> = node.movements().collect();

    if maximizing_player == node.current_player {
        let mut best_value = i8::MIN;
        let mut best_child = None;

        for child in movements {
            let (_, child_value) = alpha_beta(&node.play(&child), depth - 1, alpha, beta, maximizing_player);
            if child_value > alpha {
                alpha = child_value;
                best_child = Some(child);
                best_value = child_value;
            }
            if alpha >= beta {
                break;
            }
        }

        return (best_child, best_value);
    } else {
        let mut best_value = i8::MAX;
        let mut best_child = None;

        for child in movements {
            let (_, child_value) = alpha_beta(&node.play(&child), depth - 1, alpha, beta, maximizing_player);
            if child_value < beta {
                beta = child_value;
                best_child = Some(child);
                best_value = child_value;
            }
            if alpha >= beta {
                break;
            }
        }

        return (best_child, best_value);
    }
}

fn alpha_beta_fonc(
    node: &Configuration,
    depth: u8,
    alpha: i8,
    beta: i8,
    maximizing_player: bool,
) -> (Option<Movement>, i8) {
    if depth == 0 || node.movements().next().is_none() {
        if maximizing_player == node.current_player {
            return (None, -node.value());
        } else {
            return (None, node.value());
        };
    }

    let movements: Vec<Movement> = node.movements().collect();

    if maximizing_player == node.current_player {
        movements
            .iter()
            .map(|child| {
                let (_, child_value) = alpha_beta_fonc(&node.play(child), depth - 1, alpha, beta, maximizing_player);
                (Some(child.clone()), child_value)
            })
            .max_by_key(|&(_, value)| value)
            .unwrap_or((None, i8::MIN))
    } else {
        movements
            .iter()
            .map(|child| {
                let (_, child_value) = alpha_beta_fonc(&node.play(child), depth - 1, alpha, beta, maximizing_player);
                (Some(child.clone()), child_value)
            })
            .min_by_key(|&(_, value)| value)
            .unwrap_or((None, i8::MAX))
    }
}


impl Strategy for AlphaBeta {
    fn compute_next_move(&mut self, state: &Configuration) -> Option<Movement> {
        alpha_beta_par(state, self.0 - 1, i8::MAX, i8::MIN, state.current_player).0
    }
}

impl fmt::Display for AlphaBeta {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Alpha - Beta (max level: {})", self.0)
    }
}

/// Anytime alpha beta algorithm.
/// Any time algorithms will compute until a deadline is hit and the process is killed.
/// They are therefore run in another process and communicate through shared memory.
/// This function is intended to be called from blobwar_iterative_deepening.
pub fn alpha_beta_anytime(state: &Configuration) {
    let mut movement = AtomicMove::connect().expect("failed connecting to shmem");
    for depth in 1..100 {
        let chosen_movement = AlphaBeta(depth).compute_next_move(state);
        movement.store(chosen_movement);
    }
}
