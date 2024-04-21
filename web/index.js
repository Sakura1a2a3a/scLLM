window.addEventListener("DOMContentLoaded", function(event) {
    let submit_btn = this.document.getElementById("submit");
    let mode_btn_1 = this.document.getElementById("mode-1");
    let mode_btn_2 = this.document.getElementById("mode-2");

    mode_btn_1.style.backgroundColor = "#9BB2F8";
    mode_btn_1.style.borderColor = "#9BB2F8";
    mode_btn_1.style.color = "#790101";
    mode_btn_2.style.backgroundColor = "#790101";
    mode_btn_2.style.borderColor = "#9BB2F8";
    mode_btn_2.style.color = "#9BB2F8";

    let address_input = this.document.getElementById("select-file");
    let bs_input = this.document.getElementById("batch-size");
    let default_checker = this.document.getElementById("default-model");
    let model_input = this.document.getElementById("select-model");
    let output = this.document.getElementById("output-title");
    let sample_input = this.document.getElementById("sample");
    let lr_input = this.document.getElementById("learning-rate");
    let epoch_input = this.document.getElementById("num-epoch");

    let report_output = this.document.getElementById("report");
    let result_output = this.document.getElementById("output");

    let model_div = this.document.getElementById("model-div");
    let add_div = this.document.getElementById("add-param");

    let address = "";
    let bs = 1
    let model = "";
    let title = ""
    let sample = null;

    let is_valid = false;

    let is_prediction = true;

    let lr = 0.1;
    let num_epoch = 1;

    let lr_options = [0.1, 0.01, 0.001];

    address_input.addEventListener("click", function(event) {
        eel.getFile()(
            r => address = r
        );
    })

    model_input.addEventListener("click", function(event) {
        eel.getModel()(
            r => model = r
        );
    })

    mode_btn_1.addEventListener("click", function(event) {
        let output_div = document.getElementById("output-div");
        is_prediction = true;
        add_div.style.display = "none";
        submit_btn.innerHTML = "Start Prediction";
        output_div.style.display = "block";
        mode_btn_1.style.backgroundColor = "#9BB2F8";
        mode_btn_1.style.borderColor = "#9BB2F8";
        mode_btn_1.style.color = "#790101";
        mode_btn_2.style.backgroundColor = "#790101";
        mode_btn_2.style.borderColor = "#9BB2F8";
        mode_btn_2.style.color = "#9BB2F8";
        document.getElementById("default-check").style.display = "flex";
        document.getElementById("model-div").style.display = "flex";
        document.getElementById("output-p").innerHTML = "Prediction Result File Name: "
    })

    mode_btn_2.addEventListener("click", function(event) {
        let output_div = document.getElementById("output-div"); 
        is_prediction = false;
        add_div.style.display = "flex";
        submit_btn.innerHTML = "Get Model";
        output_div.style.display = "none";
        mode_btn_1.style.backgroundColor = "#790101";
        mode_btn_1.style.borderColor = "#9BB2F8";
        mode_btn_1.style.color = "#9BB2F8";
        mode_btn_2.style.backgroundColor = "#9BB2F8";
        mode_btn_2.style.borderColor = "#9BB2F8";
        mode_btn_2.style.color = "#790101";
        document.getElementById("default-check").style.display = "none";
        document.getElementById("model-div").style.display = "none";
        document.getElementById("output-p").innerHTML = "Model Output File Name: "
    })

    default_checker.addEventListener("click", function(event) {
        if (default_checker.checked == true){
            model_div.style.display = "none";
        } else {
            model_div.style.display = "flex";
        }
    })

    submit_btn.addEventListener("click", async function(event) {
        sample = sample_input.value;

        if ((address == "none" || address == "") && (sample == "none" || sample == null)){
            alert("Please provide a file address or select a sample dataset.");
        } else if (bs_input.value <= 0) {
            alert("Batch size needs to be bigger than 0.")
        } else if (default_checker.checked == false && model == "") {
            alert("Please select a model.")
        } else {
            bs = parseInt(bs_input.value);
            bs = (bs == "none"? 1 : bs);
            title = output.value;
            title = (title == ""? "Untitled" : title);
            if (default_checker.checked == true){
                model = ""
            }
            if (address == null || address == "none" || address == ""){
                address = sample;
            }
            if (is_prediction) {
                title += ".csv"
                await eel.predict(address, bs, model, title);
                is_valid = true;
            } else {
                title += ".pth"
                lr = lr_options[lr_input.selectedIndex]
                num_epoch = parseInt(epoch_input.value);
                await eel.get_model(address, bs, lr, num_epoch, title);
                is_valid = true;
            }
            
        }

        if (is_valid) {
            eel.get_data()(async function(result) {
                result_output.innerText = result
            }) 
        }    
    });

    this.document.getElementById("clear-input-1").addEventListener("click", function(event) {
        address = ""
        sample_input.selectedIndex = 0;
    })

    this.document.getElementById("clear-input-2").addEventListener("click", function(event) {
        bs_input.value = 1;
        output.value = "Untitled";
    })

    this.document.getElementById("clear-input-3").addEventListener("click", function(event) {
        lr_input.value = "0.1";
        epoch_input.value = 1;
    })
});
