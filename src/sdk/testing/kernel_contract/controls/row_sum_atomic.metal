const uint t = thread_position_in_grid.x;
atomic_fetch_add_explicit(&y[t / K], x[t], memory_order_relaxed);