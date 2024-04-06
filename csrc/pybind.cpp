#include "ops.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    //my custom ops
    pybind11::module ops = m.def_submodule("ops", "my custom operators");
    ops.def(
        "add",
        &torch_launch_add,
        "custom add cuda op");

}