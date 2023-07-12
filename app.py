import gradio as gr
import os
import yaml
from scripts.stable_txt2img import main
from webuiParamClass import DictI, DictT
import random


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
    opt.gpu = int(args[9])
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
    opt.fixed_code = args[26]
    opt.no_preview = args[27]


    if opt.init_img == '':
        opt.init_img = None
    if opt.seed == -1: 
        opt.seed = random.randint(0,100000)
        with open('webui-setting-config.yaml', 'r') as f:
            settings = yaml.safe_load(f)
        settings['seed'] = opt.seed
        with open('webui-setting-config.yaml','w') as f:
            yaml.dump(settings, f)
            print(f'saved latest seed = {opt.seed} to webui-setting-config.yaml')
        
  
    print(opt)
    img = main(opt)
    return img


# def random_num_gen():
#     #generate random number 
#     print("Generating random number")
#     #change seed slider value 
#     # return random.randint(0,9999999)
#     current_random_num = random.randint(0,9999999)
#     return current_random_num
    

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
                embedding_path_entries = os.listdir('subject-models/ada-0709')
                embedding_path_entries = ['subject-models/ada-0709/' + path + '/embeddings_gs-4500.pt' for path in embedding_path_entries]
                embedding_paths = gr.Dropdown (list(embedding_path_entries), label='Embedding Path', info='will add more later', multiselect=True)
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
                    # gpu = gr.Checkbox(label="Use GPU", value=True)
                    gpu = gr.Dropdown( [0,1,2,3,4,-1], label="GPU", value = '0' , info='GPU id; -1 for CPU')
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
                        seed = gr.Number(minimum=-1, maximum=100000, value=-1, label="Seed", step=1,interactive=True)
                        # seed.change(compare,[seed,random_num],seed)
                        random_num = gr.Number(value = -1, visible = False)
                    with gr.Column(scale = 0.1):
                        random_seed = gr.Button(value="🎲")
                        random_seed.style(size='sm')
                        random_seed.click(lambda x: x, random_num, seed)
                    # with gr.Column(scale=0.1):
                        previous_seed = gr.Button(value="♻️")
                        previous_seed.style(size='sm')
                        stored_seed = gr.Number(value = opt.seed,visible = False)
                        previous_seed.click(lambda x: x, stored_seed,seed)
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
                    
                    fixed_code = gr.Checkbox(label="Fixed Code")
                    no_preview = gr.Checkbox(label="No Preview")
        
               # add 'subject-models/' to everyting in the list

            with gr.Column(scale=1.5) as col2:
                #add image output 
                with gr.Row():
                    button1 = gr.Button(value="Generate")
                # button2 = gr.Button(value="Close")
                with gr.Row():
                    output = gr.Image(label="Output")
                button1.click(generate, outputs=output, inputs=[prompt, class_prompt, config, model, scale, n_iters, ddim_eta, n_samples, dim_step, gpu, embedding_paths, H, W, C, f, bs, n_repeat, n_rows, seed, precision, subj_scale, ada_emb_weight, mask_weight, broad_class, clip_last_layer_skip_weight, plmse, fixed_code, no_preview])
                # button2.click(close)
    with gr.Tab(label="Training") as tab0:
        gr.Markdown('work in progress')
        pass
    with gr.Tab(label="Settings"):
        gr.Markdown('work in progress')
        with open('webui-setting-config.yaml','r') as config_settings:
            settings = yaml.safe_load(config_settings)
        def apply_settings(*args):
            # change the setting values and save in a file so that the next time the app is opened, the settings are the same
            # settings['skip_save'] =  args[0]
            # settings['file_format'] = args[1]
            # settings['image_file_pattern'] = args[2]
            # settings['add_image_number'] = args[3]
            # settings['skip_grid'] =  args[4]
            # settings['file_format_grid'] = args[5]
            # settings['entended_info'] = args[6]
            # settings['from_file'] = args[7]
            # settings['init_img'] = args[8]
            # settings['calc_face_sim']= args[9]
            # settings['compare_with']= args[10]
            # settings['laion400m'] = args[11]

            with open('webui-setting-config.yaml','w') as f:
                yaml.dump(settings, f)
                print('saved settings to webui-setting-config.yaml')

        with gr.Row():
            with gr.Column(scale=6):
                settings_submit = gr.Button(value="Apply settings", variant='primary', elem_id="settings_submit")
            with gr.Column():
                restart_gradio = gr.Button(value='Reload UI', variant='primary', elem_id="settings_restart_gradio")

        with gr.TabItem("Saving images/grids", id="saving", elem_id="settings_tab_saving"):
            settings['skip_save'] = gr.Checkbox(label="Skip to save images", info="Save images",value= settings['skip_save'])
            settings['file_format'] = gr.Text(label="File format", info="File format to save images in",value = setting['file_format'])
            settings['image_file_pattern'] = gr.Text(label="Image file pattern", info="File pattern to save images with",value = settings['image_file_pattern'])
            settings['add_image_number']= gr.Checkbox(label="Add image number to file name", info="Add image number to file name", value = settings['add_image_number'])
            settings['skip_grid'] = gr.Checkbox(label="Skip to save image grid", info="Save image grid",value= settings['skip_grid'])
            settings['file_format_grid'] = gr.Text(label="File format", info="File format to save image grid in",value = settings['file_format_grid'] )
            settings['entended_info'] = gr.Checkbox(label="Add extended info (seed,prompt) to file name",value = settings['entended_info'])

        with gr.TabItem("Image paths", id="image_paths", elem_id="settings_tab_image_paths"):
            settings['from_file'] = gr.Textbox(lines=1, label="From File", info="The file path to load the model from",value = settings['from_file'])
            settings['init_img'] = gr.Textbox(lines=1, label="Initial Image", info="The file path to load the initial image from", value = settings['init_img'])

        with gr.TabItem("datasets", id="datasets"):
            settings['laion400m'] = gr.Checkbox(label="Laion400m",value = settings['laion400m'])
            settings['ref_prompt'] = gr.Textbox(lines=1, label="Reference Prompt", info = 'a class-level reference prompt to be mixed with the subject prompt', value = settings['ref_prompt'])

        with gr.TabItem("comparisons", id="comparisons", elem_id="settings_tab_comparisons"):
            settings['calc_face_sim'] = gr.Checkbox(label="Calc Face Sim",value= settings['calc_face_sim'])
            settings['compare_with'] = gr.Textbox(lines=1, label="Compare With", info="Enter a file path...",value = settings['compare_with'])
            settings['scores_csv'] = gr.Textbox(lines=1, label="Scores CSV", info="CSV file to save the evaluation scores", value = settings['scores_csv'])
        settings_submit.click(apply_settings)


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
    demo.launch(share=True, show_error = True)
    # demo.launch()
