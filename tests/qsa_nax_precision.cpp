// Run from the repository root; links the same libmlx as the server.
// clang++ -std=c++20 -O2 -mmacosx-version-min=26.2 tests/qsa_nax_precision.cpp -Ilib/mlx/include -Llib/mlx/lib -lmlx -o /tmp/qsa-precision
// DYLD_LIBRARY_PATH="$PWD/lib/mlx/lib" /tmp/qsa-precision
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

#include <cmath>
#include <numeric>

void replace_all(std::string& s,const std::string& a,const std::string& b) {
  size_t p=0;while((p=s.find(a,p))!=std::string::npos){s.replace(p,a.size(),b);p+=b.size();}
}
std::string zig_metal(const std::string& name) {
  auto file=read_file("src/transformer.zig");
  auto start=file.find("const "+name+" =");
  if(start==std::string::npos)throw std::runtime_error("missing kernel "+name);
  auto end=file.find("\n;",start);
  std::istringstream lines(file.substr(start,end-start));
  std::string line,result;
  while(std::getline(lines,line)) {
    auto p=line.find("\\\\");
    if(p!=std::string::npos)result+=line.substr(p+2)+"\n";
  }
  return result;
}
// CPU double reference computes sparse attention directly, independent of the GPU
// fragments, online softmax, and MLX's reduced-precision matmul implementation.
int main() {
  for(auto cfg:std::vector<std::vector<int>>{{2,17,17,512},{1,40,101,6},{1,65,65599,512}}) {
    int B=cfg[0],S=cfg[1],KV=cfg[2],KB=cfg[3];
    std::mt19937 rng(4289+S);std::vector<int> ids(size_t(B)*S*KB,INT_MAX);
    for(int b=0;b<B;b++)for(int s=0;s<S;s++) {
      int complete=(KV-S+s+1)/4;std::vector<int> choices(complete);
      std::iota(choices.begin(),choices.end(),0);std::shuffle(choices.begin(),choices.end(),rng);
      choices.resize(std::min(KB,complete));std::sort(choices.begin(),choices.end());
      std::copy(choices.begin(),choices.end(),ids.begin()+(b*S+s)*KB);
    }
    // Transposed physical layout exercises head/query/KV strides, as used by the cache.
    auto q=mx::transpose(random_array({B,S,24,256},rng),{0,2,1,3});
    auto k=mx::transpose(random_array({B,KV,2,256},rng),{0,2,1,3});
    auto v=mx::transpose(random_array({B,KV,2,256},rng),{0,2,1,3});
    auto blocks=mx::array(ids.data(),{B,S,KB});
    auto qc=mx::contiguous(mx::astype(q,mx::float32));
    auto kc=mx::contiguous(mx::astype(k,mx::float32));
    auto vc=mx::contiguous(mx::astype(v,mx::float32));mx::eval(qc,kc,vc);
    const float *Q=qc.data<float>(),*K=kc.data<float>(),*V=vc.data<float>();
    std::vector<size_t> indices;std::vector<double> reference;
    for(int b=0;b<B;b++)for(int s:std::vector<int>{0,1,S-2,S-1})for(int h:std::vector<int>{0,7,11,12,23}) {
      std::vector<int> positions;int p=KV-S+s,complete=(p+1)/4;
      for(int j=0;j<std::min(KB,complete);j++)for(int r=0;r<4;r++)positions.push_back(ids[(b*S+s)*KB+j]*4+r);
      for(int r=complete*4;r<=p;r++)positions.push_back(r);
      std::vector<double> scores;double m=-INFINITY;
      for(int pos:positions) {
        double dot=0;for(int d=0;d<256;d++)dot+=double(Q[((b*24+h)*S+s)*256+d])*K[((b*2+h/12)*KV+pos)*256+d];
        scores.push_back(dot/16);m=std::max(m,dot/16);
      }
      double z=0;for(double& a:scores){a=std::exp(a-m);z+=a;}
      for(int d=0;d<256;d++) {
        double y=0;for(size_t j=0;j<positions.size();j++)y+=scores[j]*V[((b*2+h/12)*KV+positions[j])*256+d];
        indices.push_back(((b*24+h)*S+s)*256+d);reference.push_back(y/z);
      }
    }
    for(int variant:{0,3}) {
      auto source=variant==3?read_file("src/kernels/qsa_nax.metal"):zig_metal("ATTN_QSA256_KERNEL_SOURCE");
      auto header=variant==3?read_file("src/kernels/qsa_nax_header.metal"):zig_metal("ATTN256_KERNEL_HEADER")+zig_metal("ATTN_QSA256_KERNEL_HEADER");
      replace_all(source,"device T* Op=out","device float* Op=out");
      replace_all(source,"device T* Optr = out","device float* Optr = out");
      replace_all(source,"T(Ofrag[id].x * inv)","(Ofrag[id].x * inv)");
      replace_all(source,"T(Ofrag[id].y * inv)","(Ofrag[id].y * inv)");
      auto kernel=mx::fast::metal_kernel("qsa_validate_"+std::to_string(variant),{"q","k","v","scl","blocks"},{"out"},source,header,false);
      auto out=kernel({q,k,v,mx::array({0.0625f}),blocks},{{B,24,S,256}},{mx::float32},{S*32,4,B},{32,2,1},{{"T",mx::bfloat16},{"NSG",2},{"BK",32},{"RATIO",4}},{},false,mx::Device::gpu);mx::eval(out);
      double maxerr=0,ss=0;auto y=out[0].data<float>();
      for(size_t j=0;j<indices.size();j++) {
        double e=std::abs(y[indices[j]]-reference[j]);if(!std::isfinite(e))throw std::runtime_error("nonfinite output");
        maxerr=std::max(maxerr,e);ss+=e*e;
      }
      std::cout<<"{\"variant\":"<<variant<<",\"batch\":"<<B<<",\"s\":"<<S<<",\"kv\":"<<KV<<",\"kb\":"<<KB<<",\"checked\":"<<indices.size()<<",\"max_error_vs_f64\":"<<maxerr<<",\"rmse_vs_f64\":"<<std::sqrt(ss/indices.size())<<"}"<<std::endl;
      if(maxerr>2e-6)throw std::runtime_error("FP32 output accuracy regression");
    }
  }
}
