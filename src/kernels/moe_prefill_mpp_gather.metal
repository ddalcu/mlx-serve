uint i=thread_position_in_grid.x,rows=order_shape[0],vectors=uint(x_shape[1])/8u;
if(i>=(rows+64u)*vectors)return;
uint row=i/vectors,col=i%vectors;
uint4 value=uint4(0);
if(row<rows)value=((const device uint4*)x)[(size_t)(order[row]/10u)*vectors+col];
((device uint4*)out)[i]=value;
