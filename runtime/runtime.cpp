#include "../include/macaw/runtime.hpp"
namespace macaw {
__always_inline FloatArray operator*(Matrix weight,FloatArray parameters){
    //res=weight*parameter=matrix*vector
    FloatArray res;
    float tmp;
    for (auto& x: weight) {
        for (std::size_t i = 0; i < x.size(); i++) {
            tmp+=(x[i] * parameters[i]);
        }
        res.push_back(tmp);
        tmp=0;
    }
    return res;
}
__always_inline FloatArray operator-(FloatArray x,FloatArray bias){
    //y=bias
    //just subtract each bias from each item of x to get activation
    FloatArray res;
    for (std::size_t i = 0; i < x.size(); i++) {
        res.push_back(x[i]-bias[i]);
    }
    return res;
}
__always_inline FloatArray ReLu(FloatArray activation){
    //y=max(0,x)
    FloatArray res;
    for (std::size_t i = 0; i < activation.size(); i++) {
        res.push_back(activation[i]>0?activation[i]:0);
    }
    return res;
}
Runtime::Runtime() {}
Runtime::Runtime(NeuralNetwork net) {m_network = net;}
Runtime::Runtime(NeuralNetwork net,FloatArray params) {
    m_parameters=params;
    m_network = net;
}
FloatArray Runtime::execute(FloatArray params){
    m_parameters=params;
    return execute();
}

//Real core of the class
FloatArray Runtime::execute(){
    FloatArray res;
    auto param=m_parameters;
    for(int i=0;i<m_network.size();i++){
        //output=ReLu(weight*param+bias)
        res=ReLu((m_network[i].m_weights*param)-m_network[i].m_biases);
        param=res;
    }
    return res;
}
}  // namespace macaw