//! Expose the complete FeynKit Python API in one community namespace, with
//! symbolic one-loop reduction under `hep.oneloop`.

use pyo3::{
    Bound, PyResult, Python,
    types::{PyAnyMethods, PyDictMethods, PyModule, PyModuleMethods, PyType},
};
use symbolica::api::python::SymbolicaCommunityModule;

pub struct HepModule;

impl SymbolicaCommunityModule for HepModule {
    fn get_name() -> String {
        "hep".to_owned()
    }

    fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
        feynkit_py::initialize_feynkit(module)?;
        // Reuse the upstream classes themselves, including their methods and
        // exception hierarchy, while making introspection point to our public API.
        for value in module.dict().values() {
            if value.is_instance_of::<PyType>()
                && value.getattr("__module__")?.extract::<String>()?
                    == "symbolica.community.feynkit"
            {
                value.setattr("__module__", "symbolica.community.hep")?;
            }
        }
        register_oneloop(module)?;
        Ok(())
    }

    fn initialize(py: Python<'_>) -> PyResult<()> {
        feynkit_py::FeynkitModule::initialize(py)?;
        // Registers `oneloopreduce::dot` with its `Symmetric, Linear` attributes
        // before user code can mention it and fix them to the defaults.
        oneloopreduce_python::CommunityModule::initialize(py)
    }
}

/// Add the one-loop reducer as the `symbolica.community.hep.oneloop` submodule,
/// the module name its classes declare.
fn register_oneloop(hep: &Bound<'_, PyModule>) -> PyResult<()> {
    let name = "symbolica.community.hep.oneloop";
    let oneloop = PyModule::new(hep.py(), name)?;
    oneloopreduce_python::CommunityModule::register_module(&oneloop)?;
    hep.add("oneloop", &oneloop)?;
    hep.py()
        .import("sys")?
        .getattr("modules")?
        .set_item(name, &oneloop)?;
    Ok(())
}
