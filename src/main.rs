use openvino::Core;
use openvino::DeviceType;
use openvino::ElementType;
use openvino::Tensor;
use std::error::Error;
use std::convert::TryInto;
use std::time::Instant; // Import Instant for timing

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize the OpenVINO Core interface.
    let mut core = Core::new()?;

    // Specify paths to your model files (update these paths as necessary).
    let model_xml = "model/model.xml";
    let model_bin = "model/model.bin";

    // Read the network from the IR files.
    let model = core.read_model_from_file(model_xml, model_bin)?;

    // Load the network onto the CPU. You can change "CPU" to another device if supported.
    let mut compiled_model = core.compile_model(&model, DeviceType::CPU)?;

    let input_node = compiled_model.get_input().unwrap();
    let input_node_shape = input_node.get_shape().unwrap();
    let dim1 = input_node_shape.get_dimensions();
    println!("Input Data Shape: {:?}", dim1);
    let total_elems: i64 = dim1.iter().product();
    let total_elems_usize: usize = total_elems
        .try_into()
        .expect("total_elems must be non-negative and fit into a usize");
    let input_data: Vec<f32> = vec![1.0; total_elems_usize];

    let mut tensor = Tensor::new(ElementType::F32, &input_node_shape)?;
    tensor.get_data_mut()?.copy_from_slice(&input_data);

    let mut infer_req = compiled_model.create_infer_request().unwrap();

    let _ = infer_req.set_input_tensor(&tensor);

    let start_time = Instant::now();
    let _ = infer_req.infer();
    // Compute the elapsed time.
    let elapsed_time = start_time.elapsed();
    println!("Inference took: {:?}", elapsed_time);

    let output_tensor = infer_req.get_output_tensor().unwrap();

    let output_node_shape = output_tensor.get_shape().unwrap();

    let dim2 = output_node_shape.get_dimensions();
    println!("Output Data Shape: {:?}", dim2);

    Ok(())
}

