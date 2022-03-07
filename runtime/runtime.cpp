#include "../include/macaw/runtime.hpp"
#include <iostream>
namespace macaw {

Runtime::Runtime() {}
Runtime::Runtime(NeuralNetwork net) {m_network = net;}
Runtime::Runtime(NeuralNetwork net,FloatArray params) {
    m_parameters=params;
    m_network = net;
    execute();
}
FloatArray Runtime::execute(FloatArray params){
    m_parameters=params;
    return execute();
}

//Real core of the class
FloatArray Runtime::execute(){
    FloatArray res;
    auto param=m_parameters;
    float tmp=0;
    for(int i=0;i<m_network.size();i++){
        //output=ReLu(weight*param-bias)
        auto label=m_network[i];
        for (std::size_t i = 0; i < label.m_weights.size(); i++) {
            auto& x=label.m_weights[i];
            for (std::size_t i = 0; i < x.size(); i++) {
                tmp+=(x[i] * param[i]);
            }
            tmp-=label.m_biases[i];
            res.push_back(tmp>0?tmp:0);
            tmp=0;
        }
        param=res;
    }
    return res;
}
}  // namespace macaw