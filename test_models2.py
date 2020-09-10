# Usage - python test_models.py --backup_output_dir /home/amogh,,,, --dataset taskonomy --folder_models ""

import os
import glob
import torch
import argparse
import datetime,time
import json
import numpy as np

from learning.dataloader import get_loader, get_info
from learning.mtask_grad import mtask_forone_grad
from learning.mtask_grad import mtask_forone_advacc

from models.DRNSeg import DRNSeg
from models.taskonomy_models import resnet18_taskonomy, resnet50_taskonomy

from learning.utils_learn import accuracy
from eval import eval_adv, test_selected_class_grad

def get_model(model_name, args):
    """
    Returns the model based on model name
    :param model_name:
    :return:
    """

    if model_name == 'res18' or model_name == 'resnet18':
        model = resnet18_taskonomy(pretrained=False, tasks=args.task_set)
        model = torch.nn.DataParallel(model)
    elif model_name == 'res50' or model_name == 'resnet50':
        model = resnet50_taskonomy(pretrained=False, tasks=args.task_set)
        model = torch.nn.DataParallel(model)

    elif (model_name.startswith('drn')):
        # model = DRNSeg(model_name, args.classes, pretrained_model=None, pretrained=False)
        from models.DRNSegDepth import DRNSegDepth
        model = DRNSegDepth(args.arch,
                            classes=19,
                            pretrained_model=None,
                            pretrained=False,
                            tasks=args.task_set)
        model = torch.nn.DataParallel(model)

    return model

def get_submodel_ensemble(model_name, args, task_set):
    """
    Returns the model based on model name
    :param model_name:
    :return:
    """

    if model_name == 'res18' or model_name == 'resnet18':
        model = resnet18_taskonomy(pretrained=False, tasks=task_set)
        model = torch.nn.DataParallel(model)
    elif model_name == 'res50' or model_name == 'resnet50':
        model = resnet50_taskonomy(pretrained=False, tasks=task_set)
        model = torch.nn.DataParallel(model)

    elif (model_name.startswith('drn')):
        # model = DRNSeg(model_name, args.classes, pretrained_model=None, pretrained=False)
        from models.DRNSegDepth import DRNSegDepth
        model = DRNSegDepth(args.arch,
                            classes=19,
                            pretrained_model=None,
                            pretrained=False,
                            tasks=task_set, old_version=True)
        model = torch.nn.DataParallel(model)

    return model




def test_saved_models(path_folder_models, model, val_loader,args):

    # Load the folders and the list of paths of models
    list_path_models = glob.glob(path_folder_models+"/*.tar")

    # Make the experiment summary folder
    path_folder_experiment_summary = os.path.join(args.backup_output_dir,'test_summary')
    if not os.path.exists(path_folder_experiment_summary):
        os.makedirs(path_folder_experiment_summary)

    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

    dict_summary={}
    dict_args = vars(args)
    dict_summary['config'] = dict_args
    dict_summary['results'] = {}

    # Load each model and run necessary evaluation functions
    for path_model in list_path_models:

        #TODO: if you only want to evaluate one model checkpoint
        # tmp = path_model.split('_')
        # if '100.pth.tar' not in tmp:
        #     continue

        dict_model_summary = {}

        print("=> Loading checkpoint '{}'".format(path_model))
        if torch.cuda.is_available():
            checkpoint_model = torch.load(path_model)
        else:
            checkpoint_model = torch.load(path_model,map_location=lambda storage, loc: storage)
        start_epoch = checkpoint_model['epoch']
        arch = checkpoint_model['epoch']
        # best_prec = checkpoint_model['best_prec']
        model.load_state_dict(checkpoint_model['state_dict']) #, strict=False

        info = get_info(args.dataset)
        epoch = args.epoch

        if args.dataset == 'taskonomy':
            # mtask_forone_grad → returns the avg gradient for that task during validation.
            from models.mtask_losses import get_losses_and_tasks
            taskonomy_loss, losses, criteria, taskonomy_tasks = get_losses_and_tasks(args)
            criteria_final = {'Loss': taskonomy_loss}
            for key, value in criteria.items():
                criteria_final[key] = value
            criterion = criteria_final
            grad = mtask_forone_grad(val_loader, model, criterion, args.test_task_set, args)

            print("Done with mtask_forone_grad")
            advacc_result = mtask_forone_advacc(val_loader, model, criterion, args.test_task_set, args, info, epoch,test_flag=True)

        elif args.dataset == 'cityscape':

            # Get the grad for experiment in cityscape
            grad = test_selected_class_grad(val_loader, model, args.classes, args,test_flag=True)

            print("\n\nGRAD DONE\n\n")
            # Get the advacc for cityscape
            advacc_result = eval_adv(val_loader,
                                     model,
                                     args.classes,
                                     args=args,
                                     info=info,
                                     eval_score=accuracy,
                                     calculate_specified_only=True,
                                     test_flag=True
                                     ) # Ask Chengzhi...


        dict_model_summary['grad'] = grad
        dict_model_summary['advacc'] = advacc_result
        # print(dict_model_summary['grad'],type(dict_model_summary['grad']))
        dict_summary['results'][path_model] = dict_model_summary
        # break
        # show_loss_plot(dict_summary)

        # Write dict_summary as json.
        path_summary_json = "summary_" + args.arch + "_" + args.dataset + "_" + timestamp + '.json'
        path_summary_json = os.path.join(path_folder_experiment_summary, path_summary_json)
        with open(path_summary_json, 'w') as fp:
            json.dump(dict_summary, fp, indent=4, separators=(',', ': '), sort_keys=True)
            fp.write('\n')
        print("json Dumped at", path_summary_json)

    print("END OF EXPERIMENT, Summary file written", path_summary_json)

def show_loss_plot (dict_summary):
    """Plots the dict_summary"""
    dict_results = dict_summary["results"]
    grad_list = []
    adv_acc_list = []
    # model_name_list
    for key_model, value_model in dict_results.items():
        grad_list.append(dict_results[key_model]['grad'])
        adv_acc_list.append(dict_results[key_model]['advacc'])

    import matplotlib.pyplot as plt
    f,axarr = plt.subplots(2,1,figsize=(10,10))

    # Plot grad
    # print(grad_list)

    # Plot loss
    dict_losstype_to_losslist = {}
    num_losses = len(adv_acc_list[0].keys())
    for type in list(adv_acc_list[0].keys()):
        dict_losstype_to_losslist[type] = []

    # The following for loop populates the dict_losstype_to_losslist
    for adv_acc in adv_acc_list:
        for loss_type in adv_acc.keys():
            dict_losstype_to_losslist[loss_type].append(adv_acc[loss_type])

    # Now that the loss lists for each loss type has been populated, put each of them on the second plot.
    for loss_type in dict_losstype_to_losslist.keys():
        if loss_type != 'segmentsemantic': #TODO: plot segmentsemantic as well
            list_losses = dict_losstype_to_losslist[loss_type]
            x_1 = range(len(list_losses))
            y_1 = list_losses
            axarr[1].plot(x_1,y_1,marker='o')
            for i, l in enumerate(y_1):
                axarr[1].annotate(np.round(l,2), (x_1[i], y_1[i]))

    x_0 = range(len(grad_list))
    y_0 = grad_list
    axarr[0].plot(x_0,y_0,marker='o')
    for i, l in enumerate(y_0):
        axarr[0].annotate(np.round(l,2), (x_0[i], y_0[i]))
    # Set legends and titles
    axarr[0].set(title='grad')
    axarr[1].set(title='advacc')

    # plt.legend(list(dict_losstype_to_losslist.keys()), loc='upper left')
    # plt.show()
    return f,axarr

def plot_from_file(path_result_summary):
    with open(path_result_summary) as handle:
        summary = json.loads(handle.read())
    # print((summary['results'].keys()))
    f,axarr = show_loss_plot(summary)
    # return f,axarr

def plot_from_filepath_list(list_path_result_summary,legend_grad_pre=[]):

    import matplotlib.pyplot as plt
    num_files = len(list_path_result_summary)

    f, axarr = plt.subplots(4, 1, figsize=(20, 20))
    legend_grad = []
    print("\n",len(list_path_result_summary),"\n")
    for path_summary in list_path_result_summary:
        with open(path_summary) as handle:
            summary = json.loads(handle.read())
        dict_results = summary["results"]
        grad_list = []
        adv_acc_list = []
        model_list = []
        # model_name_list
        # print("\n","appending",path_summary , "\n")
        legend_grad.append(os.path.basename(path_summary))
        for key_model, value_model in dict_results.items():
            grad_list.append(dict_results[key_model]['grad'])
            adv_acc_list.append(dict_results[key_model]['advacc'])
            model_list.append(key_model)

        # Plot loss
        dict_losstype_to_losslist = {}
        num_losses = len(adv_acc_list[0].keys())
        for type in list(adv_acc_list[0].keys()):
            dict_losstype_to_losslist[type] = []

        # The following for loop populates the dict_losstype_to_losslist
        for adv_acc in adv_acc_list:
            for loss_type in adv_acc.keys():
                dict_losstype_to_losslist[loss_type].append(adv_acc[loss_type])

        list_model_description = []
        for model_path in model_list:
            model_description = os.path.basename(model_path).split('.')[0].split("_")[1]
            try:
                model_description = int(model_description)
            except:
                pass
            list_model_description.append(model_description)

        # Now that the loss lists for each loss type has been populated, put each of them on the second plot.
            #####CODE TO plot other losses
        for loss_type in dict_losstype_to_losslist.keys():
            if loss_type != 'segmentsemantic':  # TODO: plot segmentsemantic as well
                list_losses = dict_losstype_to_losslist[loss_type]
                # x_1 = range(len(list_losses))

                # y_1 = list_losses
                # if loss_type == 'Loss':
                #     axarr[1].plot(list_model_description, y_1, marker='o')
                #     for i, l in enumerate(y_1):
                #         axarr[1].annotate(np.round(l, 2), (list_model_description[i], y_1[i]))
                # else:
                #     axarr[2].plot(list_model_description, y_1, marker='o')
                #     for i, l in enumerate(y_1):
                #         axarr[2].annotate(np.round(l, 2), (list_model_description[i], y_1[i]))
                pass

            elif loss_type == 'segmentsemantic':
                # print("segmentsemantic____", dict_losstype_to_losslist[loss_type])
                list_iou = [l["iou"] for l in dict_losstype_to_losslist[loss_type]]
                list_loss = [l["loss"] for l in dict_losstype_to_losslist[loss_type]]
                list_seg_acc = [l["seg_acc"] for l in dict_losstype_to_losslist[loss_type]]
                axarr[1].plot(list_model_description, list_iou, marker='o')
                axarr[2].plot(list_model_description, list_loss, marker='o')
                axarr[3].plot(list_model_description, list_seg_acc, marker='o')
                # for i, l in enumerate(list_iou):
                #     axarr[2].annotate(np.round(l, 2), (list_model_description[i], list_iou[i]))


        y_0 = grad_list
        axarr[0].plot(list_model_description, y_0, marker='o')
        for i, l in enumerate(y_0):
            axarr[0].annotate(np.round(l, 2), (list_model_description[i], y_0[i]))
        # Set legends and titles
        axarr[0].set(title='grad')
        # axarr[1].set(title='advacc')
        # axarr[2].set(title='advacc')
        axarr[1].set(title='segment -  iou')
        axarr[2].set(title='segment -  loss')
        axarr[3].set(title='segment - seg_acc')
        # axarr[].set(title='segment - iou, loss')

        # Setting legend
        # print("\nlegend_grad\n ", len(legend_grad), legend_grad)
        if legend_grad_pre == []:
            pass
        else:
            legend_grad = legend_grad_pre
        axarr[0].legend(legend_grad, loc='upper left')
        axarr[1].legend(legend_grad, loc='upper left')
        axarr[2].legend(legend_grad, loc='upper left')
        axarr[3].legend(legend_grad, loc='upper left')
        # axarr[1].legend(list(dict_losstype_to_losslist.keys()), loc='upper left')
        # axarr[3].legend(list(['seg_acc']), loc='upper left')
        # axarr[4].legend(list(['iou','loss']), loc='upper left')

        # axarr[0].legend(list(dict_losstype_to_losslist.keys()), loc='upper right')
        # axarr[1].legend(list(dict_losstype_to_losslist.keys()), loc='upper left')

    # axarr[0].legend([os.path.basename(l) for l in list_path_result_summary], loc='upper left')
    # axarr[1].legend([os.path.basename(l) for l in list_path_result_summary], loc='upper left')
    # axarr[2].legend([os.path.basename(l) for l in list_path_result_summary], loc='upper left')
    # axarr[3].legend([],loc='upper left')

    plt.show()


def run_test(args):
    """
    Uses the following arguments:
    args.backup folder - folder path of the format "train_res18_taskonomy_2019-10-10_15:54:09"
    args.dataset - name of the dataset
    args.arch - Architecture of the model
    :param args:
    :return:
    """

    # Read the argument values
    model_name = args.arch
    dataset = args.dataset
    experiment_backup_folder = args.backup_output_dir # Backup Folder - save logs, stats and TensorBoard logging here.

    # Get model, val_loader
    model = get_model(model_name, args)
    if torch.cuda.is_available():
        model.cuda()

    val_loader = get_loader(args, split='val', out_name=True)

    # Define the path where models reside
    path_folder_models = experiment_backup_folder + "/savecheckpoint/"

    test_saved_models(path_folder_models, model, val_loader,args)

def parse_args(backup_output_dir,
                      dataset,
                      arch,
                      data_dir,
                      task_set,
                      test_task_set,
                      step_size,
                      epoch,
                      test_batch_size,
                      classes,
                      epsilon,
                      workers,
                      pixel_scale,
                      steps,
                      debug,
                    # cityscape specific arguments
                      select_class,
                      train_category,
                      others_id
                      ):
    parser = argparse.ArgumentParser(description='Run Experiments with Checkpoint Models')
    #TODO: set required and default - ask Chengzhi
    parser.add_argument('--backup_output_dir', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--arch', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--task_set', nargs='+',default=[])
    parser.add_argument('--test_task_set', nargs='+',default=[])
    parser.add_argument('--step_size', type=int)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--test_batch_size',type=int)
    parser.add_argument('--classes',type=int)
    parser.add_argument('--epsilon',type=float)
    parser.add_argument('--workers',type=int)
    parser.add_argument('--pixel_scale',type=int)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--debug', action='store_true')
    # cityscape specific arguments
    # select_class
    args = parser.parse_args()

    args.backup_output_dir = backup_output_dir
    args.dataset = dataset
    args.arch = arch
    args.data_dir = data_dir

    task_set_whole_hard_code_sequence = []
    if 'segmentsemantic' in task_set:
        task_set_whole_hard_code_sequence.append("segmentsemantic")
    if 'depth_zbuffer' in task_set:
        task_set_whole_hard_code_sequence.append("depth_zbuffer")
    if 'edge_texture' in task_set:
        task_set_whole_hard_code_sequence.append("edge_texture")
    if 'keypoints2d' in task_set:
        task_set_whole_hard_code_sequence.append("keypoints2d")
    if 'normal' in task_set:
        task_set_whole_hard_code_sequence.append("normal")
    if 'reshading' in task_set:
        task_set_whole_hard_code_sequence.append("reshading")


    args.task_set = task_set_whole_hard_code_sequence
    args.test_task_set = test_task_set
    # args.step_size = step_size
    args.epoch = epoch
    args.test_batch_size = test_batch_size
    args.classes = classes
    # args.epsilon = epsilon
    args.workers = workers
    args.pixel_scale = pixel_scale
    args.steps = steps
    args.debug = debug

    args.epsilon = epsilon * 1.0 / pixel_scale
    args.step_size = step_size * 1.0 / pixel_scale  #TODO: bug before
    print("PRINTING ARGUMENTS \n")

    # cityscape specific arguments
    args.select_class = select_class
    args.train_category = train_category
    args.others_id = others_id

    #TODO: Manually added parameters -> read from config file
    #cityscape
    args.random_rotate = 0
    args.random_scale = 0
    args.crop_size = 896
    args.list_dir=None


    for k, v in args.__dict__.items(): # Prints arguments and contents of config file
        print(k, ':', v)
    return args

def eval_model_list(backup_output_dir,
                      dataset,
                      arch,
                      data_dir,
                      task_set,
                      test_task_set,
                      step_size,
                      epoch,
                      test_batch_size,
                      classes,
                      epsilon,
                      workers,
                      pixel_scale,
                      steps,
                      debug,
                    # cityscape specific arguments
                      select_class = None,
                      train_category = None,
                      others_id = None
                      ):

    args = parse_args(backup_output_dir,
                      dataset,
                      arch,
                      data_dir,
                      task_set,
                      test_task_set,
                      step_size,
                      epoch,
                      test_batch_size,
                      classes,
                      epsilon,
                      workers,
                      pixel_scale,
                      steps,
                      debug,
                      # cityscape specific arguments
                      select_class,
                      train_category,
                      others_id
                      )
    run_test(args)


def test_ensemble(path_model_list, model_name, task_set_whole_list, test_task_set, test_batch_size, steps, debug,
                        epsilon, step_size, dataset="taskonomy", default_suffix="/savecheckpoint/checkpoint_150.pth.tar",
                  use_noise=True, momentum=False, use_houdini=False):

    print('task_set_whole_list', task_set_whole_list)
    print('test_task_set', test_task_set)

    for i, each in enumerate(path_model_list):
        path_model_list[i] = each + default_suffix

    parser = argparse.ArgumentParser(description='Run Experiments with Checkpoint Models')
    args = parser.parse_args()

    args.dataset = dataset
    args.arch = model_name
    args.use_noise = use_noise
    args.momentum = momentum

    import socket, json
    config_file_path = "config/{}_{}_config.json".format(args.arch, args.dataset)
    with open(config_file_path) as config_file:
        config = json.load(config_file)
    if socket.gethostname() == "deep":
        args.data_dir = config['data-dir_deep']
    elif socket.gethostname() == 'hulk':
        args.data_dir = '/local/rcs/ECCV/Cityscape/cityscape_dataset'
    else:
        args.data_dir = config['data-dir']

    args.task_set = task_set_whole_list
    args.test_task_set = test_task_set
    args.test_batch_size = test_batch_size
    args.classes = config['classes']
    args.workers = config['workers']
    args.pixel_scale = config['pixel_scale']
    args.steps = steps
    args.debug = debug

    args.epsilon = epsilon
    args.step_size = step_size

    # ADDED FOR CITYSCAPES
    args.random_scale = config['random-scale']
    args.random_rotate = config['random-rotate']
    args.crop_size = config['crop-size']
    args.list_dir = config['list-dir']

    num_being_tested = len(test_task_set)

    print("PRINTING ARGUMENTS \n")
    for k, v in args.__dict__.items():  # Prints arguments and contents of config file
        print(k, ':', v)

    dict_args = vars(args)
    dict_summary = {}
    dict_summary['config'] = dict_args
    dict_summary['results'] = {}
    dict_model_summary = {}

    model_list = []
    criteria_list = []
    task_list_set =[]
    for each, path_model in zip(task_set_whole_list, path_model_list):
        model = get_submodel_ensemble(model_name, args, each)
        if torch.cuda.is_available():
            model.cuda()

        print("=> Loading checkpoint '{}'".format(path_model))
        if torch.cuda.is_available():
            checkpoint_model = torch.load(path_model)
        else:
            checkpoint_model = torch.load(path_model, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint_model['state_dict'])  # , strict=False
        model_list.append(model)

        from models.mtask_losses import get_losses_and_tasks

        print("each ", each)
        criteria, taskonomy_tasks = get_losses_and_tasks(args, customized_task_set=each)
        print("criteria got", criteria)
        criteria_list.append(criteria)
        task_list_set.extend(taskonomy_tasks)

    task_list_set = list(set(task_list_set))
    # print('dataloader will load these tasks', task_list_set)
    val_loader = get_loader(args, split='val', out_name=False, customized_task_set=task_list_set)

    from models.ensemble import Ensemble
    model_whole = Ensemble(model_list)

    from learning.test_ensemble import mtask_ensemble_test

    # mtask_ensemble_test(val_loader, model_ensemble, criterion_list, task_name, args, info)

    # print('mid test task', args.test_task_set)
    advacc_result = mtask_ensemble_test(val_loader, model_whole, criteria_list, args.test_task_set, args, use_houdini=use_houdini)
    print("Results: epsilon {} step {} step_size {}  Acc for task {} ::".format(args.epsilon, args.steps,
                                                                                args.step_size, args.test_task_set),
          advacc_result)


def test_one_checkpoint(path_model, model_name, task_set_whole, test_task_set, test_batch_size, steps, debug,
                        epsilon, step_size, output_dir='./', norm_type='Linf', dataset="taskonomy"):

    parser = argparse.ArgumentParser(description='Run Experiments with Checkpoint Models')
    args = parser.parse_args()

    args.dataset = dataset
    args.arch = model_name

    import socket, json
    config_file_path = "config/{}_{}_config.json".format(args.arch, args.dataset)
    with open(config_file_path) as config_file:
        config = json.load(config_file)
    if socket.gethostname() == "deep":
        args.data_dir = config['data-dir_deep']
    elif socket.gethostname() == "amogh":
        args.data_dir = config['data-dir_amogh']
    elif socket.gethostname() == 'hulk':
        args.data_dir = '/local/rcs/ECCV/Taskonomy/taskonomy-sample-model-1-small-master/'
    else:
        args.data_dir = config['data-dir']

    args.task_set = task_set_whole
    args.test_task_set = test_task_set
    # args.step_size = step_size
    args.test_batch_size = test_batch_size
    args.classes = config['classes']
    # args.epsilon = epsilon
    args.workers = config['workers']
    args.pixel_scale = config['pixel_scale']
    args.steps = steps
    args.debug = debug

    args.epsilon = epsilon
    args.step_size = step_size

    # ADDED FOR CITYSCAPES
    args.random_scale = config['random-scale']
    args.random_rotate = config['random-rotate']
    args.crop_size = config['crop-size']
    args.list_dir = config['list-dir']


    num_being_tested = len(test_task_set)

    print("PRINTING ARGUMENTS \n")
    for k, v in args.__dict__.items():  # Prints arguments and contents of config file
        print(k, ':', v)


    dict_args = vars(args)
    dict_summary = {}
    dict_summary['config'] = dict_args
    dict_summary['results'] = {}
    dict_model_summary = {}

    model_checkpoint_name = path_model.split('/')
    path_folder_experiment_summary = os.path.join(output_dir, 'test_summary'+ model_checkpoint_name[0])
    if not os.path.exists(path_folder_experiment_summary):
        os.makedirs(path_folder_experiment_summary)

    model = get_model(model_name, args)
    if torch.cuda.is_available():
        model.cuda()

    val_loader = get_loader(args, split='val', out_name=False)



    print("=> Loading checkpoint '{}'".format(path_model))
    if torch.cuda.is_available():
        checkpoint_model = torch.load(path_model)
    else:
        checkpoint_model = torch.load(path_model, map_location=lambda storage, loc: storage)
    start_epoch = checkpoint_model['epoch']
    epoch = checkpoint_model['epoch']
    # best_prec = checkpoint_model['best_prec']
    model.load_state_dict(checkpoint_model['state_dict'])  # , strict=False

    print('epoch is {}'.format(epoch))

    # Initialise the data structures in which we are going to save the statistics
    # Assign the variables that would be used ny eval function
    # Mtask_forone_grad → returns the avg gradient for that task during validation.

    from models.mtask_losses import get_losses_and_tasks
    # taskonomy_loss, losses, criteria, taskonomy_tasks = get_losses_and_tasks(args)
    criteria, taskonomy_tasks = get_losses_and_tasks(args)

    info = get_info(args.dataset)

    # mtask_forone_advacc	 → Calculates the IoU but does not return it.
    from learning.mtask_validate import mtask_test_clean
    advacc_result = mtask_forone_advacc(val_loader, model, criteria, args.test_task_set, args, info, epoch,
                                        test_flag=True, norm=norm_type)


    dict_model_summary['advacc'] = advacc_result
    dict_summary['results'][path_model] = dict_model_summary
    # break
    # show_loss_plot(dict_summary)
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

    path_summary_json = "summary_" + args.arch + "_" + args.dataset + "_" + timestamp + '.json'
    path_summary_json = os.path.join(path_folder_experiment_summary, path_summary_json)
    with open(path_summary_json, 'w') as fp:
        json.dump(dict_summary, fp, indent=4, separators=(',', ': '), sort_keys=True)
        fp.write('\n')
    print("json Dumped at", path_summary_json)

    return dict_model_summary


def revert(in_list):
    out_list = []
    for each in range(len(in_list)):
        out_list.append(in_list[-(each+1)])
    return out_list


if __name__ == '__main__':
    # eval_model_list('tasknoy',,,,)
    # eval_model_list('tasknoy',,,,) # time and date,
    # eval_model_list('tasknoy',,,,)
    # eval_model_list('tasknoy',,,,)

    pass
