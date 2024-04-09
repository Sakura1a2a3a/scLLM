from ultil import predict_method, further_tune

while True:

    print("Welcome to scLLM, we are a large sc-language model to help you do cell-type annotation")
    print("Please select one of the following functions: ")
    print("1: Make prediction")
    print("2: Further tune your own model")
    print("3: Quit")


    choice = input(" ")

    if choice == "1":
        dataset = input("dataset: \n ")
        batch_size = input("batch size: \n ")
        model_path = input("model_path: \n ")
        out_path = input("out_path: \n ")

        
        # dataset = './data/demo.h5ad', batch_size = 6, model_path = './model/default_state_dict.pth', out_path = "./predicted_results.csv"
        predict_method(dataset, int(batch_size), model_path, out_path)
        

    if choice == "2":
        dataset = input("dataset: \n ")
        batch_size = input("batch size: \n ")
        learning_rate = input("learning_rate: \n ")
        num_epochs = input("num_epochs: \n ")
        model_path = input("model_path: \n ")

        # dataset = './data/demo.h5ad', batch_size = 6, learning_rate = 0.001, num_epochs = 1, model_path = './model/further_tune_state_dict.pth'
        further_tune(dataset, int(batch_size), float(learning_rate), int(num_epochs), model_path)
        

        

    if choice == "3":
        print("Thank you for using scLLM")
        break


