//! Lock helpers for state shared across Python threads.
//!
//! Compute-heavy bindings run inside `Python::detach`, so the GIL no longer
//! serialises access to native state. Mutable state therefore lives behind a
//! `Mutex`/`RwLock`, and every acquisition happens *inside* the detached
//! closure: a thread never waits on a lock while holding the GIL, and never
//! needs the GIL while holding a lock, so the two cannot deadlock.
//!
//! A panic inside a binding surfaces in Python as `PanicException` and
//! poisons the lock it held. The guarded values stay structurally valid (a
//! panic cannot leave a Rust value half-moved), so the helpers recover the
//! guard instead of turning every later call into a second panic.

use std::sync::{Mutex, MutexGuard, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

pub(crate) fn read<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(PoisonError::into_inner)
}

pub(crate) fn write<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(PoisonError::into_inner)
}

pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}
