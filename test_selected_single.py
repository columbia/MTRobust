DEBUG=False
import socket
import os

if socket.gethostname() == 'deep':
    root_path ="/mnt/md0/2019Fall/SegSaveLog/cityscape/multi-task-small/drn-22/visualize_model/"
elif socket.gethostname() == 'hulk':
    root_path = "/local/rcs/ECCV/Cityscape/Houdini/lambda0.01"
    # root_path = "/local/rcs/mcz/MultiTask/cityscape/multi-task-small/drn-22/visualize_model/"
single = ["adv_s-only-lam0.01", "adv_s-sd-lam0.01", "adv_s-sA-lam0.01", "adv_s-sdA-lam0.01"]
# single = ["trainset_s_testset_s_lambda_0_2", "trainset_sd_testset_s_lambda_0.001",
#           "trainset_sA_testset_s_lambda_0.1", "trainset_sdA_testset_s_lambda_0.001"]
    # list_name = [["segmentsemantic"] for i in range(9)]

list_name = [["segmentsemantic"], ["segmentsemantic", "depth_zbuffer"],
             ["segmentsemantic", "autoencoder"], ["segmentsemantic", "depth_zbuffer", "autoencoder"]]



model_path_list = []
for i, each in enumerate(single):
    model_path_list.append(os.path.join(root_path, each))
print("root", root_path)

from test_models2 import *

# for ii in range(3,4):
#     print("\n\n\nindividually FGSM")
#     test_ensemble(
#         [model_path_list[ii]],
#         "drn_d_22",
#         [list_name[ii]],
#         ["segmentsemantic"],
#         test_batch_size=2,
#         steps=1,
#         debug=DEBUG,
#         epsilon=8,
#         step_size=8,
#         dataset="cityscape", default_suffix="/savecheckpoint/checkpoint_200.pth.tar", use_noise=False)

# Rebuttal, increasing number of steps for PGD attacks
# Notice though using a general test ensemble function, but the attack is for single model.
batch_size = 4
# for ii in range(4):
#     num_steps = 10
#     print("\n\n\nindividually PGD, num steps {}".format(num_steps))
#     test_ensemble(
#         [model_path_list[ii]],
#         "drn_d_22",
#         [list_name[ii]],
#         ["segmentsemantic"],
#         test_batch_size=batch_size,
#         steps=num_steps,
#         debug=DEBUG,
#         epsilon=8,
#         step_size=2,
#         dataset="cityscape", default_suffix="/savecheckpoint/checkpoint_200.pth.tar", use_noise=False)
#
# for ii in range(4):
#     num_steps = 20
#     print("\n\n\nindividually PGD, num steps {}".format(num_steps))
#     test_ensemble(
#         [model_path_list[ii]],
#         "drn_d_22",
#         [list_name[ii]],
#         ["segmentsemantic"],
#         test_batch_size=batch_size,
#         steps=num_steps,
#         debug=DEBUG,
#         epsilon=8,
#         step_size=2,
#         dataset="cityscape", default_suffix="/savecheckpoint/checkpoint_200.pth.tar", use_noise=False)
#
#
# for ii in range(4):
#     num_steps = 50
#     print("\n\n\nindividually PGD, num steps {}".format(num_steps))
#     test_ensemble(
#         [model_path_list[ii]],
#         "drn_d_22",
#         [list_name[ii]],
#         ["segmentsemantic"],
#         test_batch_size=batch_size,
#         steps=num_steps,
#         debug=DEBUG,
#         epsilon=8,
#         step_size=2,
#         dataset="cityscape", default_suffix="/savecheckpoint/checkpoint_200.pth.tar", use_noise=False)
#
# for ii in range(4):
#     num_steps = 100
#     print("\n\n\nindividually PGD, num steps {}".format(num_steps))
#     test_ensemble(
#         [model_path_list[ii]],
#         "drn_d_22",
#         [list_name[ii]],
#         ["segmentsemantic"],
#         test_batch_size=batch_size,
#         steps=num_steps,
#         debug=DEBUG,
#         epsilon=8,
#         step_size=2,
#         dataset="cityscape", default_suffix="/savecheckpoint/checkpoint_200.pth.tar", use_noise=False)


# for ii in range(4):
#     num_steps = 3
#     print("\n\n\nindividually BIM, num steps {}".format(num_steps))
#     test_ensemble(
#         [model_path_list[ii]],
#         "drn_d_22",
#         [list_name[ii]],
#         ["segmentsemantic"],
#         test_batch_size=batch_size,
#         steps=num_steps,
#         debug=DEBUG,
#         epsilon=8,
#         step_size=2,
#         dataset="cityscape", default_suffix="/savecheckpoint/checkpoint_200.pth.tar", use_noise=False)

# for ii in range(3,4):
#     print("\n\n\nindividually BIM")
#     test_ensemble(
#         [model_path_list[ii]],
#         "drn_d_22",
#         [list_name[ii]],
#         ["segmentsemantic"],
#         test_batch_size=2,
#         steps=3,
#         debug=DEBUG,
#         epsilon=8,
#         step_size=2,
#         dataset="cityscape", default_suffix="/savecheckpoint/checkpoint_200.pth.tar", use_noise=False)

for ii in range(4):
    # print("\n\n\nindividually MIM")
    num_steps = 20
    print("\n\n\nindividually Houdini Attack, num steps {}".format(num_steps))
    test_ensemble(
        [model_path_list[ii]],
        "drn_d_22",
        [list_name[ii]],
        ["segmentsemantic"],
        test_batch_size=batch_size,
        steps= num_steps,
        debug=DEBUG,
        epsilon=4,
        step_size=1,
        dataset="cityscape", default_suffix="/savecheckpoint/checkpoint_200.pth.tar", use_noise=True, momentum=True,
        use_houdini=True)


exit(0)

if socket.gethostname() == 'deep':
    root_path ="/mnt/md0/2019Fall/SegSaveLog/cityscape/multi-task-small/drn-22-d-related/visualize_model/"
elif socket.gethostname() == 'hulk':
    root_path = "/local/rcs/mcz/MultiTask/cityscape/multi-task-small/drn-22-d-related/visualize_model/"


single = ["trainset_d_testset_d_lambda_0.1_seed_227_lrs_140_200",
"trainset_sd_testset_d_lambda_0.1_seed_42_lrs_140_200",
          "trainset_dA_testset_d_lambda_0.1_seed_42_lrs_140_200",
"trainset_sdA_testset_d_lambda_0.01_seed_42_lrs_140_200"]

    # list_name = [["segmentsemantic"] for i in range(9)]

list_name = [["depth_zbuffer"], ["segmentsemantic", "depth_zbuffer"],
             ["depth_zbuffer", "autoencoder"], ["segmentsemantic", "depth_zbuffer", "autoencoder"]]


model_path_list = []
for i, each in enumerate(single):
    model_path_list.append(root_path + each)
print("root", root_path)

from test_models2 import *

# for ii in range(3,4):
#     print("\n\n\nindividually FGSM")
#     test_ensemble(
#         [model_path_list[ii]],
#         "drn_d_22",
#         [list_name[ii]],
#         ["depth_zbuffer"],
#         test_batch_size=2,
#         steps=1,
#         debug=DEBUG,
#         epsilon=8,
#         step_size=8,
#         dataset="cityscape", default_suffix="/savecheckpoint/checkpoint_200.pth.tar", use_noise=False)

###################
######--Rebuttal PGD more steps
batch_size = 32
# for ii in range(4):
#     step_num = 10
#     print("\n\n\nindividually PGD step={}".format(step_num))
#     test_ensemble(
#         [model_path_list[ii]],
#         "drn_d_22",
#         [list_name[ii]],
#         ["depth_zbuffer"],
#         test_batch_size=batch_size,
#         steps=step_num,
#         debug=DEBUG,
#         epsilon=8,
#         step_size=2,
#         dataset="cityscape", default_suffix="/savecheckpoint/checkpoint_200.pth.tar", use_noise=True)
# for ii in range(4):
#     step_num = 20
#     print("\n\n\nindividually PGD step={}".format(step_num))
#     test_ensemble(
#         [model_path_list[ii]],
#         "drn_d_22",
#         [list_name[ii]],
#         ["depth_zbuffer"],
#         test_batch_size=batch_size,
#         steps=step_num,
#         debug=DEBUG,
#         epsilon=8,
#         step_size=2,
#         dataset="cityscape", default_suffix="/savecheckpoint/checkpoint_200.pth.tar", use_noise=True)
#
#
# for ii in range(4):
#     step_num = 50
#     print("\n\n\nindividually PGD step={}".format(step_num))
#     test_ensemble(
#         [model_path_list[ii]],
#         "drn_d_22",
#         [list_name[ii]],
#         ["depth_zbuffer"],
#         test_batch_size=batch_size,
#         steps=step_num,
#         debug=DEBUG,
#         epsilon=8,
#         step_size=2,
#         dataset="cityscape", default_suffix="/savecheckpoint/checkpoint_200.pth.tar", use_noise=True)
#
# for ii in range(4):
#     step_num = 100
#     print("\n\n\nindividually PGD step={}".format(step_num))
#     test_ensemble(
#         [model_path_list[ii]],
#         "drn_d_22",
#         [list_name[ii]],
#         ["depth_zbuffer"],
#         test_batch_size=batch_size,
#         steps=step_num,
#         debug=DEBUG,
#         epsilon=8,
#         step_size=2,
#         dataset="cityscape", default_suffix="/savecheckpoint/checkpoint_200.pth.tar", use_noise=True)

##############


# for ii in range(4):
#     step_num = 3
#     print("\n\n\nindividually BIM step=".format(step_num))
#     test_ensemble(
#         [model_path_list[ii]],
#         "drn_d_22",
#         [list_name[ii]],
#         ["depth_zbuffer"],
#         test_batch_size=2,
#         steps=step_num,
#         debug=DEBUG,
#         epsilon=8,
#         step_size=2,
#         dataset="cityscape", default_suffix="/savecheckpoint/checkpoint_200.pth.tar", use_noise=False)

for ii in range(4):
    step_num=100
    print("\n\n\nindividually MIM step=".format(step_num))
    test_ensemble(
        [model_path_list[ii]],
        "drn_d_22",
        [list_name[ii]],
        ["depth_zbuffer"],
        test_batch_size=2,
        steps= step_num,
        debug=DEBUG,
        epsilon=8,
        step_size=2,
        dataset="cityscape", default_suffix="/savecheckpoint/checkpoint_200.pth.tar", use_noise=True, momentum=True,
        use_houdini=True)

