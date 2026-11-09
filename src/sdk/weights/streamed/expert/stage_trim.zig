//! A wide prompt call's staged records its routes do not take let go before the drive reaches them (the
//! stream's `Stream.trimStaged`).
//!
//! Staging (stage.zig) reads a wide layer's predicted transient loads into its windows ahead of its call (or during
//! the call before it); the call's routes take the staged records they route and the rest are read for nothing. So:
//! - a staged window's records are posted in jobs in rank order (the layout's, hottest first), not file order, so the
//!   jobs the call is likeliest to leave are the last posted, the likeliest still queued when its routes are known;
//! - once the call has its routes and has taken its staged groups (`Stream.trimStaged`, from the wide lane's adoption),
//!   each staged job holding none of the call's experts that no read worker has started is withdrawn from the read pool
//!   (`Pool.withdraw`: its tickets published unread, no row written), its rows let go (a reading row emptied, every row
//!   unpinned, named no longer); a job a worker has started reads as posted.
//! Exact: only reads no route of the call takes are dropped; a record is the same bytes in any row.

/// The most experts a layer has that the trim tracks (a larger bank is left untrimmed).
pub const max_experts = 512;
