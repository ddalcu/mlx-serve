#include <mlx/mlx.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
namespace mx = mlx::core;
std::string read_file(const std::string& path) {
    std::ifstream f(path); std::stringstream s; s << f.rdbuf();
    if (!f) throw std::runtime_error("Cannot read " + path);
    return s.str();
}
mx::array random_array(mx::Shape shape, std::mt19937& rng) {
    size_t n=1; for(int d:shape)n*=d;
    std::vector<float> v(n); std::uniform_real_distribution<float> dist(-1,1);
    for(auto& x:v)x=dist(rng);
    return mx::astype(mx::array(v.data(),shape),mx::bfloat16);
}

#include <numeric>
#include <set>
std::string f32store(std::string src) {
  auto p=src.find("device T* Op=out");
  if(p==std::string::npos)throw std::runtime_error("output pointer missing");
  src.replace(p,std::string("device T* Op=out").size(),"device float* Op=out");return src;
}
int main(int argc,char**) {
  std::mt19937 rng(18471);
  auto header=read_file("src/kernels/qsa_nax_header.metal");
  auto base=mx::fast::metal_kernel("pair_oracle_base",{"q","k","v","scl","blocks"},{"out"},f32store(read_file("src/kernels/qsa_nax.metal")),header,false);
  auto candidate=mx::fast::metal_kernel("pair_oracle_new",{"q","k","v","scl","blocks","tilepos","tilemask"},{"out"},f32store(read_file("src/kernels/qsa_pair.metal")),header,false);
  auto planner=mx::fast::metal_kernel("pair_oracle_plan",{"blocks","kvlen"},{"tilepos","tilemask"},read_file("src/kernels/qsa_pair_plan.metal"),"",false);
  struct C{int B,S,KV,KB;};
  auto shapes=argc>1?std::vector<C>{{1,8192,8192,512}}:std::vector<C>{{2,17,17,512},{2,65,65599,512},{2,130,159,31},{1,17,200003,512},{1,17,1048576,512},{1,8192,65536,512},{1,8703,65536,512}};
  for(auto sh:shapes) {
    auto [B,S,KV,KB]=sh;int NG=(S+1)/2,NT=2*((KB+1)*4+31)/32+3;
    std::vector<int> ids(size_t(B)*S*KB,2147483647);
    for(int b=0;b<B;b++)for(int s=0;s<S;s++) {
      int complete=(KV-S+s+1)/4,n=std::min(complete,KB);
      std::vector<int> all(complete);std::iota(all.begin(),all.end(),0);
      // Alternate overlap-heavy and independent selections within pairs.
      std::mt19937 local(b*871+((s/4)%2?s:s/2));std::shuffle(all.begin(),all.end(),local);
      all.resize(n);std::sort(all.begin(),all.end());
      std::copy(all.begin(),all.end(),ids.begin()+(b*S+s)*KB);
    }
    auto q=mx::transpose(random_array({B,S,24,256},rng),{0,2,1,3});
    auto k=mx::transpose(random_array({B,KV,2,256},rng),{0,2,1,3});
    auto v=mx::transpose(random_array({B,KV,2,256},rng),{0,2,1,3});
    auto blocks=mx::array(ids.data(),{B,S,KB}),scl=mx::array({.0625f});
    auto plan=planner({blocks,mx::array(KV)},{{B*NG,NT,8},{B*NG,NT}},{mx::int32,mx::int32},{NG*32,1,B},{32,1,1},{{"KB",KB},{"RATIO",4},{"NT",NT}},{},false,mx::Device::gpu);
    auto ref=base({q,k,v,scl,blocks},{{B,24,S,256}},{mx::float32},{S*32,4,B},{32,2,1},{{"T",mx::bfloat16},{"BK",32},{"NSG",2},{"RATIO",4}},{},false,mx::Device::gpu)[0];
    auto out=candidate({q,k,v,scl,blocks,plan[0],plan[1]},{{B,24,S,256}},{mx::float32},{NG*32,8,B},{32,4,1},{{"T",mx::bfloat16},{"BK",32},{"NSG",4},{"G",2},{"RATIO",4}},{},false,mx::Device::gpu)[0];
    mx::eval(out,ref,plan[0],plan[1]);
    if(!mx::all(mx::isfinite(out)).item<bool>())throw std::runtime_error("nonfinite result");
    const auto* pp=plan[0].data<int>();const auto* masks=plan[1].data<int>();
    for(int b=0;b<B;b++)for(int s=0;s<S;s++) {
      int p=KV-S+s,complete=(p+1)/4;
      std::vector<int> expect,got;
      for(int i=0;i<std::min(complete,KB);i++)for(int r=0;r<4;r++)expect.push_back(ids[(b*S+s)*KB+i]*4+r);
      for(int r=complete*4;r<=p;r++)expect.push_back(r);
      size_t gi=b*NG+s/2;
      for(int it=0;it<NT;it++)if((masks[gi*NT+it]>>(s%2))&1)
        for(int r=0;r<32;r++){int base=pp[(gi*NT+it)*8+r/4];int pos=base<0?-1:base+r%4;if(pos>=0&&pos<=p)got.push_back(pos);}
      std::sort(got.begin(),got.end());
      if(got!=expect)throw std::runtime_error("selected-key multiset changed");
    }
    auto qc=mx::contiguous(mx::astype(q,mx::float32)),kc=mx::contiguous(mx::astype(k,mx::float32)),vc=mx::contiguous(mx::astype(v,mx::float32));
    mx::eval(qc,kc,vc);const float* Q=qc.data<float>(),*K=kc.data<float>(),*V=vc.data<float>();
    const float* actual=out.data<float>(),*stock=ref.data<float>();
    double worst_new=0,worst_stock=0,worst_bf_new=0,worst_bf_stock=0;int checked=0;
    auto ob=mx::astype(mx::astype(out,mx::bfloat16),mx::float32),rb=mx::astype(mx::astype(ref,mx::bfloat16),mx::float32);mx::eval(ob,rb);
    const float* actual_b=ob.data<float>(),*stock_b=rb.data<float>();
    for(int b=0;b<B;b++)for(int s:std::set<int>{0,1,2,std::min(S-1,2047),std::min(S-1,2048),S/2,S-2,S-1})for(int h:{0,11,12,23}) {
      int p=KV-S+s,complete=(p+1)/4;std::vector<int> pos;
      for(int i=0;i<std::min(KB,complete);i++)for(int r=0;r<4;r++)pos.push_back(ids[(b*S+s)*KB+i]*4+r);
      for(int r=complete*4;r<=p;r++)pos.push_back(r);
      std::vector<double> scores;double maximum=-1e300,den=0;
      const float* qr=Q+((b*24+h)*S+s)*256;
      for(int p0:pos) {const float* kr=K+((b*2+h/12)*KV+p0)*256;double score=0;
        for(int d=0;d<256;d++)score+=double(qr[d])*kr[d];score*=.0625;scores.push_back(score);maximum=std::max(maximum,score);}
      for(auto& z:scores){z=std::exp(z-maximum);den+=z;}
      for(int d=0;d<256;d++) {
        double want=0;for(size_t i=0;i<pos.size();i++)want+=scores[i]*V[((b*2+h/12)*KV+pos[i])*256+d];want/=den;
        size_t idx=((b*24+h)*S+s)*256+d;
        double se=std::abs(stock[idx]-want),ne=std::abs(actual[idx]-want);
        double sbe=std::abs(stock_b[idx]-want),nbe=std::abs(actual_b[idx]-want);
        worst_new=std::max(worst_new,ne);worst_stock=std::max(worst_stock,se);
        worst_bf_new=std::max(worst_bf_new,nbe);worst_bf_stock=std::max(worst_bf_stock,sbe);checked++;
        if(ne>std::max(1.5*se,2.5e-6) || nbe>std::max(1.5*sbe,2e-3))throw std::runtime_error("float64 per-element parity failed");
      }
    }
    std::cout<<"{\"B\":"<<B<<",\"S\":"<<S<<",\"KV\":"<<KV<<",\"KB\":"<<KB<<",\"coverage\":\"all rows exact\",\"f64_checked\":"<<checked<<",\"f32_stock_max\":"<<worst_stock<<",\"f32_pair_max\":"<<worst_new<<",\"bf16_stock_max\":"<<worst_bf_stock<<",\"bf16_pair_max\":"<<worst_bf_new<<",\"passed\":true}"<<std::endl;
  }
}
