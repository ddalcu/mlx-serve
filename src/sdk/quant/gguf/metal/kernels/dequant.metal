uint u = thread_position_in_grid.x;
GgOut<T> o = {out + size_t(u) * UNIT};
DEQ_AT(w, u / (UPB * NB), (u / UPB) % NB, u % UPB, o);