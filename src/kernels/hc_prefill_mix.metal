const int idx=thread_position_in_grid.x;
if(idx>=M*H)return;
const int row=idx/H,col=idx%H;
T value=T(0);
for(int h=0;h<HC;h++) {
  const int off=(row*HC+h)*H+col;
  T sig=sigtab[as_type<ushort>(up[off])];
  T product=T(float(sig)*float(normed[off]));
  value=h==0 ? product : T(float(product)+float(value));
}
out[idx]=T(float(value)/float(HC));
