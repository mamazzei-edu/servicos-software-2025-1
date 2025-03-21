import gradio as gr
import requests

def envia(imagem):
    url = "http://backend-image:9001/imagens/"
    with open(imagem, 'rb') as f:
        request = requests.post(url, files={"image_file": f})
        return request.content

ui = gr.Interface(fn=envia, inputs=gr.Image(type="filepath"), outputs="text")

if __name__ == "__main__":
    ui.launch(server_name ="0.0.0.0", server_port=7860,show_api=False)

