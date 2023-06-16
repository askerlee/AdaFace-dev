import gradio as gr
import os
import yaml
from scripts.stable_txt2img import main
from webuiParamClass import DictI, DictT
import random

current_seed = random.randint(0,9999999)
default_json = {"seed": current_seed}
stats = gr.State(value=default_json)
opt = DictI() # create an instance containing default param for Inference

with open('webui-setting-config.yaml','r') as f: # open config file to load pre settings
    setting = yaml.load(f, Loader=yaml.FullLoader)
    for key, value in setting.items():
        setattr(opt, key, value)
    print(setting)


def generate(*args):
 
    
    opt.prompt = args[0]
    opt.class_prompt = args[1]
    opt.config = args[2]
    opt.ckpt = args[3]
    opt.scale = args[4]
    opt.n_iters = args[5]
    opt.ddim_eta = args[6]
    opt.n_samples = args[7]
    opt.dim_step = args[8]
    opt.gpu = args[9]
    opt.embedding_paths = args[10]
    opt.H = args[11]
    opt.W = args[12]
    opt.C = args[13]
    opt.f = args[14]
    opt.bs = args[15]
    opt.n_repeat = args[16]
    opt.n_rows = args[17]
    opt.seed = args[18]
    opt.precision = args[19]
    opt.subj_scale = args[20]
    opt.ada_emb_weight = args[21]
    opt.mask_weight = args[22]
    opt.broad_class = args[23]
    opt.clip_last_layer_skip_weight = args[24]
    opt.plms = args[25]
    opt.laion400m = args[26]
    opt.fixed_code = args[27]


    if opt.init_img == '':
        opt.init_img = None
    if opt.gpu: 
        opt.gpu = 1
    else:
        opt.gpu = -1
  
    print(opt)
    print(model,"value")
    print(model,"info")

    img = main(opt)


    # generate code
    return img
random_num = gr.Slider(visible=False, value=-1)
def compare(seed, random_num):
    if seed != random_num:
        return -1

def random_num_gen():
    #generate random number 
    print("Generating random number")
    #change seed slider value 
    # return random.randint(0,9999999)
    current_random_num = random.randint(0,9999999)
    return current_random_num
    

with gr.Blocks() as demo:
    gr.Markdown("# AdaPrompt")
    model_entries = os.listdir('models/stable-diffusion/')
    model_entries = ['models/stable-diffusion/' + model_entry for model_entry in model_entries]
    print(model_entries)
    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            model = gr.Dropdown( model_entries, label='Model (checkpoint)',value = 'v1-5-pruned.ckpt',info = 'under models/stable-diffusion/ directory') #add value ='preious selection from state
            print(model)
            # model = 'models/stable-diffusion/' + f'{model}' 
            # print(model)
            

        with gr.Column(scale=3):
            pass
    #add a drop down
    # list of config is the item under the path, read all item under a directory
    with gr.Tab(label="Inference") as tab1:
        with gr.Row() as row0:
            with gr.Column(scale=2) as col1:
                prompt = gr.Textbox(lines=2, label="Prompt", placeholder="Enter a prompt...")
                class_prompt = gr.Textbox(lines=2, label="Class Prompt", placeholder="Enter a class prompt...")
                with gr.Row() as row00:
                    with gr.Column():
                        config_entries = os.listdir('configs/stable-diffusion/')
                        config_entries = ['configs/stable-diffusion/' + path for path in config_entries]
                        config = gr.Dropdown( list(config_entries), label='Config', value ='configs/stable-diffusion/v1-inference-ada.yaml')
                    with gr.Column():
                        n_iters = gr.Slider(minimum=0, maximum=75, value=20, label="Number of Iterations", step=1)
                with gr.Row() as row2:
                    ddim_eta = gr.Slider(minimum=0, maximum=1, value=0.0, label="Dimension", step=0.1)
                    dim_step = gr.Slider(minimum=0, maximum=250,value=50,  label="Dimension Step", step=1)
                    gpu = gr.Checkbox(label="Use GPU", value=True)
                with gr.Row() as row3:
                    with gr.Column():
                        H = gr.Slider(minimum=0, maximum=1000, value=512, label="Height", step=1)
                        W = gr.Slider(minimum=0, maximum=1000, value=512, label="Width", step=1)
                    with gr.Column():
                        n_samples = gr.Slider(minimum=0, maximum=20, value=8, label="Number of Samples", step=1)
                        n_rows = gr.Slider(minimum=0, maximum=5, value=2, label="Number of Rows", step=1)
                with gr.Row() as row:
                    scale = gr.Slider(minimum=0, maximum=20, value=10.0 ,label="Scale", step=1)
                with gr.Row() as row4:
                    C = gr.Slider(minimum=0, maximum=4, value=4, label="Channels", step=1)
                    f = gr.Slider(minimum=0, maximum=10, value=8, label="f", step=1)
                    bs = gr.Slider(minimum=0, maximum=20, value=8, label="Batch Size", step=1)
                    n_repeat = gr.Slider(minimum=0, maximum=20, value=1, label="n_repeat", step=1)
                with gr.Row() as row5:
                    with gr.Column(scale=0.8):
                        seed = gr.Slider(minimum=0, maximum=100, value=42, label="Seed", step=1,interactive=True)
                        # seed.change(compare,[seed,random_num],seed)
                        
                    # with gr.Column(scale = 0.1):
                    #     random_seed = gr.Button(value="🎲")
                    #     random_seed.style(size='sm')
                    #     # random_seed.click(random_num_gen)
                    # # with gr.Column(scale=0.1):
                    #     previous_seed = gr.Button(value="♻️")
                    #     previous_seed.style(size='sm')
                        # previous_seed.click()
                with gr.Row():
                    precision = gr.Dropdown(['autocast', 'float32', 'float64'], label='Precision',value='autocast')
                with gr.Row() as row6:
                    subj_scale = gr.Slider(minimum=0, maximum=10, value=1.0, label="Subject Scale", step=0.1)
                    ada_emb_weight = gr.Slider(minimum=-1, maximum=10, value=-1.0, label="Ada Emb Weight", step=0.1)
                    mask_weight = gr.Slider(minimum=0, maximum=10, value=0.0, label="Mask Weight", step=0.1)
                    broad_class = gr.Slider(minimum=0, maximum=10, value=0, label="Broad Class", step=1)
                    clip_last_layer_skip_weight = gr.Slider(minimum=0, maximum=10, value=0.5, label="Clip Last Layer Skip Weight", step=0.1)
                
                with gr.Row() as row7:
                    # skip_grid = gr.Checkbox(label="Skip Grid", info="Skip Grid")
                    plmse = gr.Checkbox(label="PLMSE")
                    # skip_save = gr.Checkbox(label="Skip Save", info="Skip Save")
                    laion400m = gr.Checkbox(label="Laion400m")
                    fixed_code = gr.Checkbox(label="Fixed Code")
                    no_preview = gr.Checkbox(label="No Preview")
        
                embedding_path_entries = os.listdir('subject-models/ada-delta')
                embedding_path_entries = ['subject-models/ada-delta/' + path + '/embeddings_gs-4500.pt' for path in embedding_path_entries]
                embedding_paths = gr.Dropdown (list(embedding_path_entries), label='Embedding Path', info='will add more later', multiselect=True)
                # add 'subject-models/' to everyting in the list

            with gr.Column(scale=1.5) as col2:
                #add image output 
                with gr.Row():
                    button1 = gr.Button(value="Generate")
                # button2 = gr.Button(value="Close")
                with gr.Row():
                    output = gr.Image(label="Output")
                button1.click(generate, outputs=output, inputs=[prompt, class_prompt, config, model, scale, n_iters, ddim_eta, n_samples, dim_step, gpu, embedding_paths, H, W, C, f, bs, n_repeat, n_rows, seed, precision, subj_scale, ada_emb_weight, mask_weight, broad_class, clip_last_layer_skip_weight, plmse, laion400m, fixed_code, no_preview])
                # button2.click(close)
    with gr.Tab(label="Training") as tab0:
        gr.Markdown('work in progress')
        pass
    with gr.Tab(label="Settings"):
        gr.Markdown('work in progress')
        def apply_settings(*args):
            setting = {}
            # change the setting values and save in a file so that the next time the app is opened, the settings are the same
            setting['skip_save'] =  args[0]
            setting['file_format'] = args[1]
            setting['image_file_pattern'] = args[2]
            setting['add_image_number'] = args[3]
            setting['skip_grid'] =  args[4]
            setting['file_format_grid'] = args[5]
            setting['entended_info'] = args[6]
            setting['from_file'] = args[7]
            setting['init_img'] = args[8]
            setting['calc_face_sim']= args[9]
            setting['compare_with']= args[10]

            with open('config.yaml','w') as f:
                yaml.dump(setting, f)
                print('saved settings to config.yaml')

        with gr.Row():
            with gr.Column(scale=6):
                settings_submit = gr.Button(value="Apply settings", variant='primary', elem_id="settings_submit")
            with gr.Column():
                restart_gradio = gr.Button(value='Reload UI', variant='primary', elem_id="settings_restart_gradio")

        with gr.TabItem("Saving images/grids", id="saving", elem_id="settings_tab_saving"):
            save_img = gr.Checkbox(label="Skip to save images", info="Save images",value= opt['skip_save'])
            file_format = gr.Text(label="File format", placeholder="jpg", info="File format to save images in")
            image_file_pattern = gr.Text(label="Image file pattern", placeholder="image_{i}", info="File pattern to save images with")
            add_image_number = gr.Checkbox(label="Add image number to file name", info="Add image number to file name")
            save_img_grid = gr.Checkbox(label="Skip to save image grid", info="Save image grid",value= opt['skip_grid'])
            file_format_grid = gr.Text(label="File format", placeholder="jpg", info="File format to save image grid in")
            extended_info = gr.Checkbox(label="Add extended info (seed,prompt) to file name")

        with gr.TabItem("Image paths", id="image_paths", elem_id="settings_tab_image_paths"):
            from_file = gr.Textbox(lines=1, label="From File", placeholder="Enter a file path...", info="The file path to load the model from")
            init_img = gr.Textbox(lines=1, label="Initial Image", placeholder="Enter a file path...", info="The file path to load the initial image from")


        with gr.TabItem("comparisons", id="comparisons", elem_id="settings_tab_comparisons"):
            calc_face_sim = gr.Checkbox(label="Calc Face Sim")
            compare_with = gr.Textbox(lines=1, label="Compare With", placeholder="Enter a file path...")

        settings_submit.click(apply_settings, inputs=[save_img, file_format, image_file_pattern, add_image_number, save_img_grid, file_format_grid,extended_info, from_file, init_img, calc_face_sim, compare_with])


        # with gr.TabItem("Actions", id="actions", elem_id="settings_tab_actions"):
        #         request_notifications = gr.Button(value='Request browser notifications', elem_id="request_notifications")
        #         download_localization = gr.Button(value='Download localization template', elem_id="download_localization")
        #         reload_script_bodies = gr.Button(value='Reload custom script bodies (No ui updates, No restart)', variant='secondary', elem_id="settings_reload_script_bodies")
        #         with gr.Row():
        #             unload_sd_model = gr.Button(value='Unload SD checkpoint to free VRAM', elem_id="sett_unload_sd_model")
        #             reload_sd_model = gr.Button(value='Reload the last SD checkpoint back into VRAM', elem_id="sett_reload_sd_model")
        # with gr.TabItem("Licenses", id="licenses", elem_id="settings_tab_licenses"):
        #         pass
    
    with gr.Tab(label = 'Generated Images'):
        def display_images(dir):
            #list images that ends with .jpg
            images = os.listdir(dir)
            print(dir)
            images = [dir + i  for i in images if i.endswith('.jpg')]
            print(images)
            return images
        
        imgButton = gr.Button("Images")
        imgDir = gr.Textbox(value ='./outputs/samples/', visible = False)
            #display images from outputs/samples directory

        gridButton = gr.Button("Grid")
        gridDir = gr.Textbox(value = "./outputs/", visible=False)
            #display images from outputs directory

        gallery = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery"
        ).style(columns=[2], height="auto")

        imgButton.click(display_images,imgDir,gallery)
        gridButton.click(display_images,gridDir,gallery)


    

    
    

if __name__ == "__main__":
    demo.launch(share=True)
    # demo.launch()
