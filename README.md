# Multimodal Feature Fusion via Graph Neural Networks for Visual Dialog
A graph neural network implementation using implicit and explicit relation modeling on to create feature graphs to perform multi-modal feature fusion on the Visual Dialog dataset by Das et al. (2016).

## Motivation
To solve the challenge of Visual Dialog, the implementation from Chen et al. (2021) from the paper "GoG: Relation-aware Graph-over-Graph Network for Visual Dialog." The implementation of Chen et al. (2021) serves as the baseline implementation and the novel master node architecture mentioned explained in the thesis serves as the new implementation.

## Main Packages used
1. [Light-weight Transformer for Many Inputs (LTMI)](https://github.com/davidnvq/visdial)
2. [Detectron2](https://detectron2.readthedocs.io/en/latest/)
3. [LAL-Parser](https://github.com/KhalilMrini/LAL-Parser)
4. [NeuralCoref](https://github.com/huggingface/neuralcoref), [AllenNLP (conviniently packaged)](https://demo.allennlp.org/coreference-resolution/), [e2eCoref (source)](https://github.com/kentonl/e2e-coref)
5. [NeuSpell](https://github.com/neuspell/neuspell)

## Project Structure
The directory structure can be found under the **directory_tree.txt** or **dir_files_tree.txt** under the project root. However, since the nature of the project involves the use of a lot of packages, it can become a bit difficult to understand. This section explains all the different directories on the server. If the GitLab is missing any directory, it usually means that the directory contains large embeddings or is a dependency package that can be found on GitHub.

#### Guide to Directories
> **documents** - Contains all submission related docs including declaration, proposals, thesis, presentation files.

> **data** - Contains the VisDial 1.0 and COCO 2014 dataset that can be downloaded from their respective websites. Also contains a subdirectory called **subsets** which contains the smaller VisDial train datasets that were created for faster experimentation.

> **embeddings** - Contains the embeddings generated for the questions (question embeddings, graphs, graph attention network, global attention pooling), answers (answer embeddings, graphs, graph attention network, global attention pooling), history (history embeddings, graphs, graph attention network, global attention pooling), images (image embeddings, graphs, graph attention network, global attention pooling), glove (all glove models from source, including the custom LTMI glove weights), fusion (training batch embeddings for the baseline, master node, and ablation (setting which doesn't have any semantic awareness)). 
> The questions and history directory have three subdirectories each (1, 10, 100) that contain the embeddings mentioned earlier for each subset. The images directory has 2 directories: instance and panoptic, each of which contain (1, 10, 100) directories like questions and history. Note: Unfortunately, the files and embeddings for panoptic segmentation weren't saved and therefore need to be recoded.

> **env_configs** contains contain the 4 different conda environments used to manage the different packages (neural parser (LAL Parser by Mrini et al.), detectron2 (facebook), neural coref (huggingface), and neuspell (grammar correction) all requires version crontrol). The repositories can be found online.

> **github** contains the 3 main packages - detectron2, LAL-parser, and Visdial (LTMI - Nguyen et al. 2020). Visdial is dependent on bottom-up-attention (the fourth directory under the github package). They instructions to build these packages can be found in their respective github repositories.
 
 > **notebooks** - contains the scripts to create the embeddings for the two architectures, along with the master node implementation and the gog baseline implementation.
 
 > **notebooks_val** - contains the scripts to generate the validation embeddings and graphs from visdial val. I inlcuded this in a seperate directory. Note: While working with notebooks has been really beneficial for prototyping, converting them into scripts with argparse is the next goal.
 
 > **others** - contains two notebooks which are alternate experiments i wanted to run with the allennlp coreference resolution tool and resnet101 instead of resnet50 that i run to perform the image segmentation.
 
 > **outputs** - contains a list of outputs stored from notebooks to be used for the thesis content.
 
 > **scrap** - contains test experiments to working with generating the pruned question graphs and the spatial relation modeling based pruned image graphs. Can be ignored.
 
 > **tools** - contain three files which are to store frequently used functions (under the next goal, tools will be listed as utils following standard package naming).

> **checkpoints** - contains a copy of the tensorboard checkpoints from the LTMI model under github/visdial

#### Guide to Files
The guide follows the steps taken to create the whole project. Order of appearence indicates precedence of execution, unless mentioned otherwise. The notebooks also contain a subset number which can be toggled which allows to operate between the three different subsets, i.e. 1, 10, and 100.
> Main Notebooks
	> - **notebooks/data_subset.ipnb** contains the script to create the subset data for visdial.
	> - **notebooks/history_graph.ipynb** contains the script to generate the structured dialog history graphs which includes generating the history graphs for each round of question-answering using coreference resolution. Also contains the test scripts to ensure that Global Attention Network (GAT) update and Global Attention Pooling (GAP) can be performed on the generated graphs.
	> - **notebooks/question_graph.ipynb** contains the the script for generating the question graphs from the output of the LAL parser. Also contains the test scripts to ensure that (GAT) update and (GAP) can be performed on the generated graphs.
	> - **notebooks/image_graph.ipynb** contains the script for generating the image graphs. Also contains the test scripts to ensure that GAT update and GAP can be performed on the generated graphs.
	> - **notebooks/gog.ipynb** contains the baseline implementation of Chen et al. (2021) to create the three relation aware feature graphs. The notebook explains how the embedding steps are performed in a bottom up approach: history graph -> history-aware question graph -> history-aware-question-aware image graph using GAT and GAP.
	> - **notebooks/master_node.ipynb** contains the novel architecture where instead of node value concatenation as a method of relation-awareness, a fully-connected node has been added to the upper graph: history graph -> history-aware question graph -> history-aware-question-aware image graph using GAT and GAP.
	> - **notebooks/control.ipynb** contains the script where GAT and GAP are performed on all three graphs seperately and no relation-awareness is carried out. Acts as a control study to highlight the advantages of the relation awareness performed by gog and master_node.

**Note:**
1. Since history_graphs.ipynb are generating the history graphs using coreference resolution, the conda environment to be used for this is coreference or e2e-coref. Just before the graph creation step, switch to the gnnVD environment since gnnVD was the environment where torch and torch geometric were configured (torch geometric was not installing correctly with the older python version required for the coreference tools used.
2. To generate the embeddings required for the question graphs using the LAL parser, use the neural-parser conda environment.
3. question_graph.ipynb, image_graph.ipynb, gog.ipynb, master_node.ipynb, and control.ipynb all use the root environment gnnVD (for both notebooks and notebook_val) directories.

> Validation Scripts	
	> - **notebooks_val/*** contains the same scripts as the notebooks directory for history_graph.ipynb, question_graph.ipynb, image_graph.ipynb, master_node.ipynb. I created them seperately since the directories containing the validation files were very different in structure as compared to the train files. 

> Other Notebooks
	> - **notebooks/glove.ipynb** contain the scripts used to load the different glove models.
	> - **notebooks/Glove.ipynb** contains the script with the hand-crafted values used by the LTMI model
	> - **notebooks/caption.ipynb** contains the script performing caption analysis between visdial captions and coco captions


#### Guide to using the training scripts with LTMI
> Training scripts
	> - Under the project root, there's train.ipynb, train_control.ipynb, train_gog.ipynb.
	> - train.ipynb is the master_node implementation which creates the dataloader and maps the master node embeddings manually to the LTMI dataloader (which is also the visdial dataloader) using the image_id of the LTMI dataloader as the primary key.
	> - Similarly, train_control.ipynb and train_gog.ipynb also map the control embeddings and the gog embeddings to the LTMI dataloader using the image_id as the primary key.

> Modification scripts for LTMI implementation to integrate the gog and master_node embeddings:
	> - Under the project root directory, there's also 4 files that are a copy of updated training scripts used to update the LTMI model with the embeddings generated by gog and master_node and control. This is done because LTMI is packaged to work with it's own embeddings.
	> - the new_train.py goes in the project root of the LTMI, i.e. under github/visdial/. This is used to run the model.
	> - the new_options.py contains the argparse initialization code which has been updated with the new weights gathered from Nguyen et al. (2020). The embeddings that the authors sent me were slightly different compared to the ones mentioned under their original github repo (the bottom up attention embedding) which had to be remapped. Also the validation step kept failing even with batch size 1, so I had to disable the validation step. This is also a task for the future. new_options.py also goes in the project root folder of LTMI, i.e. github/visdial/.
	> - the encoder.py script contains the remapping of the gog, maste_node and control with the input from LTMI. The embeddings are mapped to the image_id from the LTMI train dataloader which serves as the primary key. This goes under the encoder directory contained in the LTMI project root, i.e. github/visdial/encoders/.
	> - The LTMI visdial repository also packages the code from Das et. al. (2016) which can be found in github/visdial/visdial. Since this uses an older version of pytorch, the dynamic_rnn.py script had to be modified to correctly unload some operations off the gpu to the cpu and then reloaded back to the gpu. A copy of this modification exists in my project root. This can be copied to github/visdial/visdial/common/.

#### To execute the train script, run the following:
```console
python3 new_train.py --config_name model_v10 --save_dir checkpoints --batch_size 8 --decoder_type misc --init_lr 0.001 --scheduler_type "LinearLR" --num_epochs 15 --num_samples 123287 --milestone_steps 3 5 7 9 11 13 --encoder_out 'img' 'ques' --dropout 0.1 --img_has_bboxes --ca_has_layer_norm --ca_num_attn_stacks 2 --ca_has_residual --ca_has_self_attns --txt_has_layer_norm --txt_has_decoder_layer_norm --txt_has_pos_embedding
```
Different parameters can be used to change the settings for the experiments.

#### Personal Comment
As was suggested by my supervisors, instead of manually mapping everything to use with the LTMI model, using the VisDial dataloader would probably have been easier. However, at the time, I wasn't able to figure it out on how to use it. And instead of not doing anything because I was stuck, I thought it best to create my own.