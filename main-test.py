import eel
import wx
import os
from ultil import predict_method, further_tune

eel.init("web")

@eel.expose
def get_data():
    return "Data from python"

@eel.expose
def predict(address, bs, model, output):
    filename = address
    script_dir = os.path.dirname(__file__)
    if model == "":
        default_model_path = os.path.join(script_dir, "model/default_state_dict.pth")
        model = default_model_path
    output_path = os.path.join(script_dir, output)
    print("Address:", filename)
    print("Batch Size:", bs)
    print("Output Title:", output_path)
    print("Model:", model)
    predict_method(filename, bs, model, output_path)

@eel.expose
def get_model(address, bs, lr, epoch, output):
    filename = address
    script_dir = os.path.dirname(__file__)
    output_path = os.path.join(script_dir, "model/" + output)
    print("Address:", filename)
    print("Batch Size:", bs)
    print("Learning Rate:", lr)
    print("Number of Epoch:", epoch)
    print("Output Title:", output_path)
    further_tune(filename, bs, lr, epoch, output_path)

@eel.expose
def getFile(wildcard="*"):
    app = wx.App(None)
    style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
    dialog = wx.FileDialog(None, 'Open', wildcard=wildcard, style=style)
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPath()
    else:
        path = None
    dialog.Destroy()
    return path

@eel.expose
def getModel(wildcard="*"):
    app = wx.App(None)
    style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
    dialog = wx.FileDialog(None, 'Open', wildcard=wildcard, style=style)
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPath()
    else:
        path = None
    dialog.Destroy()
    return path

eel.start("index.html")