// Same eight-group BF16 product/sum order as native MLX, top-K=10.
const uint col=thread_position_in_grid.x, token=thread_position_in_grid.y;
if(col>=2560u)return;
bfloat acc=bfloat(0.0f);
for(uint group=0;group<8u;group++){
  bfloat partial=bfloat(0.0f);
  for(uint slot=group;slot<10u;slot+=8u){
    uint original=token*10u+slot;
    bfloat product=down[(size_t)inverse[original]*2560u+col]*scores[original];
    partial=product+partial;
  }
  acc=group==0?partial:partial+acc;
}
out[(size_t)token*2560u+col]=acc;
