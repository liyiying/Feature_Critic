import os
import numpy as np
import torch
from torch.autograd import Variable
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from data_process.data_gen_PACS import *
from utils import *
from networks import *
import datetime
import copy

class ModelBaseline_PACS:
    def __init__(self, flags):
        #torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.configure(flags)
        self.setup_path(flags)
        self.init_network_parameter(flags)

        if not os.path.exists(flags.logs):
            os.mkdir(flags.logs)
        if not os.path.exists(flags.model_path):
            os.mkdir(flags.model_path)

    def __del__(self):
        print('release source')

    def configure(self,flags):
        self.flags_log = os.path.join(flags.logs, '%s.txt'%(flags.method))
        self.activate_load_model = False


    def setup_path(self, flags):
        self.best_accuracy_val = -1
        if flags.dataset == 'PACS':
            self.domains_name = get_domain_name()
            data_folder, train_data, val_data, test_data = get_data_folder()
        else:
            assert flags.dataset == 'PACS', 'The current homogeneous DG code uses PACS dataset'

        self.train_paths = []
        for data in train_data:
            path = os.path.join(data_folder, data)
            self.train_paths.append(path)

        self.val_paths = []
        for data in val_data:
            path = os.path.join(data_folder, data)
            self.val_paths.append(path)

        unseen_index = flags.unseen_index
        self.unseen_data_path = []
        self.unseen_data_path.append(self.train_paths[unseen_index])
        self.unseen_data_path.append(os.path.join(data_folder, test_data[unseen_index]))
        self.train_paths.remove(self.train_paths[unseen_index])
        self.val_paths.remove(self.val_paths[unseen_index])

        if not os.path.exists(flags.logs):
            os.mkdir(flags.logs)

        flags_log = os.path.join(flags.logs, 'path_log.txt')
        write_log(str(self.train_paths), flags_log)
        write_log(str(self.val_paths), flags_log)
        write_log(str(self.unseen_data_path), flags_log)

        self.batImageGenTrains = []
        for train_path in self.train_paths:
            batImageGenTrain = BatchImageGenerator(flags=flags, file_path=train_path, stage='train',
                                                   metatest=False, b_unfold_label=False)
            self.batImageGenTrains.append(batImageGenTrain)

        self.batImageGenTrains_metatest = []
        for train_path in self.train_paths:
            batImageGenTrain_metatest = BatchImageGenerator(flags=flags, file_path=train_path, stage='train',
                                                   metatest=True, b_unfold_label=False)
            self.batImageGenTrains_metatest.append(batImageGenTrain_metatest)

        self.batImageGenVals = []
        for val_path in self.val_paths:
            batImageGenVal = BatchImageGenerator(flags=flags, file_path=val_path, stage='val',
                                                 metatest=False, b_unfold_label=True)
            self.batImageGenVals.append(batImageGenVal)

    def init_network_parameter(self,flags):

        self.weight_decay = 5e-5
        self.batch_size = flags.batch_size

        self.h = 4096
        self.hh = 512

        self.num_domain = 4
        self.num_test_domain = 1
        self.num_train_domain = self.num_domain - self.num_test_domain

        ######################################################
        self.feature_extractor_network = alexnet(pretrained=True).cuda()

        # phi means the classifer network parameter, from h (the output feature layer of input data) to c (the number of classes).
        self.para_theta = self.feature_extractor_network.parameters()
        self.phi = classifier_homo(flags.num_classes)
        self.opt_phi = torch.optim.SGD(self.phi.parameters(), lr=flags.lr, weight_decay=self.weight_decay, momentum=0.9)
        self.opt_theta = torch.optim.SGD(self.para_theta, lr=flags.lr,
                                         weight_decay=self.weight_decay, momentum=0.9)

        self.ce_loss = torch.nn.CrossEntropyLoss()

    def load_state_dict(self, state_dict=''):

        tmp = torch.load(state_dict)
        pretrained_dict = tmp[0]
        # load the new state dict
        self.feature_extractor_network.load_state_dict(pretrained_dict)
        self.phi.load_state_dict(tmp[1])

    def heldout_test(self, flags):

        # load the best model on the validation data
        model_path = os.path.join(flags.model_path, 'best_model.tar')
        self.load_state_dict(state_dict=model_path)

        batImageGenTest_test = BatchImageGenerator(flags=flags, file_path=self.unseen_data_path[1],
                                                        metatest=False, stage='test', b_unfold_label=True)
        images_test = batImageGenTest_test.images
        labels_test = batImageGenTest_test.labels

        threshold = 100
        if len(images_test) > threshold:

            n_slices_test = len(images_test) / threshold
            indices_test = []
            for per_slice in range(n_slices_test - 1):
                indices_test.append(len(images_test) * (per_slice + 1) / n_slices_test)
            test_image_splits = np.split(images_test, indices_or_sections=indices_test)
        self.feature_extractor_network.eval()
        self.phi.eval()
        # split the test data into splits and test them one by one
        test_classifier_output = []
        for test_image_split in test_image_splits:
            # print(len(test_image_split))
            test_image_split = get_image(test_image_split)
            # print (test_image_split[0].shape)
            test_image_split = torch.from_numpy(np.array(test_image_split, dtype=np.float32))
            test_image_split = Variable(test_image_split, requires_grad=False).cuda()
            feature_out = self.feature_extractor_network(test_image_split)

            classifier_out = self.phi(feature_out).data.cpu().numpy()
            test_classifier_output.append(classifier_out)
        test_classifier_output = np.concatenate(test_classifier_output)
        torch.cuda.empty_cache()
        accuracy = compute_accuracy(predictions=test_classifier_output, labels=labels_test)
        precision = np.mean(test_classifier_output == labels_test)
        print(accuracy)
        print(precision)
        # accuracy
        accuracy_info = 'the test domain %s.\n' % (flags.unseen_index)
        flags_log = os.path.join(flags.logs, 'heldout_test_log.txt')
        write_log(accuracy_info, flags_log)
        #write_log(accuracy, flags_log)
        write_log(precision, flags_log)

    def train(self, flags):

        time_start = datetime.datetime.now()

        for _ in range(flags.iteration_size):
            self.feature_extractor_network.train()

            total_loss = 0.0
            for i in range(self.num_train_domain):
                self.phi.train()
                images_train, labels_train = self.batImageGenTrains[i].get_images_labels_batch()

                x_subset = torch.from_numpy(images_train.astype(np.float32))
                y_subset = torch.from_numpy(labels_train.astype(np.int64))
                # wrap the inputs and labels in Variable
                x_subset, y_subset = Variable(x_subset, requires_grad=False).cuda(), \
                                 Variable(y_subset, requires_grad=False).long().cuda()

                y_pred = self.phi(self.feature_extractor_network(x_subset))
                loss = self.ce_loss(y_pred, y_subset)
                total_loss += loss

            self.opt_theta.zero_grad()
            self.opt_phi.zero_grad()
            total_loss.backward()
            self.opt_theta.step()
            self.opt_phi.step()

            if _ % 500 == 0 and flags.debug is True and _ != 0:
                time_end = datetime.datetime.now()
                epoch = (flags.iteration_size -int(_))/500
                time_cost = epoch*(time_end-time_start).seconds/60
                time_start = time_end
                torch.cuda.empty_cache()
                print('the number of iteration %d, and it is expected to take another %d minutes to complete..'%(_, time_cost))
                self.validate_workflow(self.batImageGenVals, flags, _)
                torch.cuda.empty_cache()

    def validate_workflow(self, batImageGenVals, flags, ite):

        accuracies = []
        for count, batImageGenVal in enumerate(batImageGenVals):
            accuracy_val = self.test(batImageGenTest=batImageGenVal, flags=flags, ite=ite,
                                     log_dir=flags.logs, log_prefix='val_index_{}'.format(count), count=count)

            accuracies.append(accuracy_val)

        mean_acc = np.mean(accuracies)

        if mean_acc > self.best_accuracy_val:
            self.best_accuracy_val = mean_acc

            f = open(os.path.join(flags.logs, 'Best_val.txt'), mode='a')
            f.write('ite:{}, best val accuracy:{}\n'.format(ite, self.best_accuracy_val))
            f.close()

            if not os.path.exists(flags.model_path):
                os.mkdir(flags.model_path)

            outfile = os.path.join(flags.model_path, 'best_model.tar')
            if flags.method == 'baseline':
                torch.save((self.feature_extractor_network.state_dict(), self.phi.state_dict()), outfile)
            if flags.method == 'Feature_Critic':
                torch.save((self.feature_extractor_network.state_dict(), self.phi.state_dict(), self.omega.state_dict()), outfile)

    def test(self, flags, ite, log_prefix, log_dir='logs/', batImageGenTest=None, count=0):

        self.feature_extractor_network.eval()
        self.phi.eval()
        if batImageGenTest is None:
            batImageGenTest = BatchImageGenerator(flags=flags, file_path='', stage='test', metatest=False, b_unfold_label=False)

        images_test = batImageGenTest.images
        labels_test = batImageGenTest.labels

        threshold = 100
        if len(images_test) > threshold:

            n_slices_test = len(images_test) / threshold
            indices_test = []
            for per_slice in range(n_slices_test - 1):
                indices_test.append(len(images_test) * (per_slice + 1) / n_slices_test)
            test_image_splits = np.split(images_test, indices_or_sections=indices_test)

            # Verify the splits are correct
            test_image_splits_2_whole = np.concatenate(test_image_splits)
            assert np.all(images_test == test_image_splits_2_whole)

            # split the test data into splits and test them one by one
            test_image_preds = []
            for test_image_split in test_image_splits:
                #print(len(test_image_split))
                test_image_split = get_image(test_image_split)
                #print (test_image_split[0].shape)
                images_test_split = torch.from_numpy(np.array(test_image_split, dtype=np.float32))
                images_test_split = Variable(images_test_split, requires_grad=False).cuda()

                classifier_out = self.phi(self.feature_extractor_network(images_test_split)).data.cpu().numpy()
                test_image_preds.append(classifier_out)

            # concatenate the test predictions first
            predictions = np.concatenate(test_image_preds)
        else:
            images_test = get_image(images_test)
            images_test = torch.from_numpy(np.array(images_test, dtype=np.float32))
            images_test = Variable(images_test, requires_grad=False).cuda()
            predictions = self.phi(self.feature_extractor_network(images_test)).data.cpu().numpy()

        accuracy = compute_accuracy(predictions=predictions, labels=labels_test)
        print('----------accuracy test of domain ----------:', accuracy)

        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        log_path = os.path.join(log_dir, '{}.txt'.format(log_prefix))
        write_log(str('ite:{}, accuracy:{}'.format(ite, accuracy)), log_path=log_path)

        return accuracy

class Model_Feature_Critic_PACS(ModelBaseline_PACS):
    def __init__(self, flags):

        ModelBaseline_PACS.__init__(self, flags)
        self.init_dg_function(flags)

    def __del__(self):
        print('release source')

    def init_dg_function(self,flags):
        self.dg_function = {'MLP': 1, 'Flatten_FTF': 2}
        self.id_dg = self.dg_function[flags.type]

        if self.id_dg == 1:
            self.omega = Critic_Network_MLP(self.h, self.hh).cuda()
        if self.id_dg == 2:
            self.omega = Critic_Network_Flatten_FTF(self.h, self.hh).cuda()

    def train(self, flags):

        write_log(flags, self.flags_log)
        self.pre_train()
        self.reinit_network_P(flags)
        time_start = datetime.datetime.now()

        for _ in range(flags.iteration_size):
            self.iteration = _
            if _ == 15000:
                self.opt_phi = torch.optim.SGD(self.phi.parameters(), lr=flags.lr/10, weight_decay=self.weight_decay,
                                               momentum=0.9)
                self.opt_theta = torch.optim.SGD(self.para_theta, lr=flags.lr/10,
                                                 weight_decay=self.weight_decay, momentum=0.9)

            self.feature_extractor_network.train()
            self.phi.train()

            index = np.random.permutation(self.num_train_domain)
            meta_train_idx = index[0:1]
            meta_test_idx = index[2:]
            write_log('-----------------iteration_%d--------------'%(_), self.flags_log)
            write_log(meta_train_idx, self.flags_log)
            write_log(meta_test_idx, self.flags_log)

            for itr in range(flags.meta_iteration_size):
                meta_train_loss_main = 0.0
                meta_train_loss_dg = 0.0
                meta_loss_held_out = 0.0

                for i in meta_train_idx:
                    domain_a_x, domain_a_y = self.batImageGenTrains[i].get_images_labels_batch()
                    x_subset_a = torch.from_numpy(domain_a_x.astype(np.float32))
                    y_subset_a = torch.from_numpy(domain_a_y.astype(np.int64))
                    x_subset_a, y_subset_a = Variable(x_subset_a, requires_grad=False).cuda(), \
                                 Variable(y_subset_a, requires_grad=False).long().cuda()

                    feat_a = self.feature_extractor_network(x_subset_a)
                    pred_a = self.phi(feat_a)

                    loss_main = self.ce_loss(pred_a, y_subset_a)
                    meta_train_loss_main += loss_main
                    if self.id_dg == 1:
                        loss_dg = self.beta * self.omega(feat_a)
                    if self.id_dg == 2:
                        loss_dg = self.beta * self.omega(torch.matmul(torch.transpose(feat_a, 0, 1), feat_a).view(1, -1))

                    meta_train_loss_dg += loss_dg

                self.opt_phi.zero_grad()
                self.opt_theta.zero_grad()
                meta_train_loss_main.backward(retain_graph=True)

                grad_theta = [theta_i.grad for theta_i in self.feature_extractor_network.parameters()]
                theta_updated_old = {}
                '''
                for (k, v), g in zip(self.feature_extractor_network.state_dict().items(),grad_theta):
                    theta_updated[k] = v - self.alpha * g
                '''
                # Todo: fix the new running_mean and running_var
                # Because Resnet18 network contains BatchNorm structure, there is no gradient in BatchNorm with running_mean and running_var.
                # Therefore, these two factors should be avoided in the calculation process of theta_old and theta_new.
                num_grad = 0
                for i, (k, v) in enumerate(self.feature_extractor_network.state_dict().items()):

                    if grad_theta[num_grad] is None:
                        num_grad +=1
                        theta_updated_old[k] = v
                    else:
                        theta_updated_old[k] = v - self.alpha * grad_theta[num_grad]
                        num_grad += 1

                meta_train_loss_dg.backward(create_graph=True)

                grad_theta = [theta_i.grad for theta_i in self.feature_extractor_network.parameters()]
                theta_updated_new = {}
                num_grad = 0
                for i, (k, v) in enumerate(self.feature_extractor_network.state_dict().items()):

                    if grad_theta[num_grad] is None:
                        num_grad +=1
                        theta_updated_new[k] = v
                    else:
                        theta_updated_new[k] = v - self.alpha * grad_theta[num_grad]
                        num_grad += 1

                temp_new_feature_extractor_network = alexnet(pretrained=False).cuda()
                fix_nn(temp_new_feature_extractor_network, theta_updated_new)
                temp_new_feature_extractor_network.train()

                temp_old_feature_extractor_network = alexnet(pretrained=False).cuda()
                temp_old_feature_extractor_network.load_state_dict(theta_updated_old)
                temp_old_feature_extractor_network.train()

                for i in meta_test_idx:
                    domain_b_x, domain_b_y = self.batImageGenTrains_metatest[i].get_images_labels_batch()
                    x_subset_b = torch.from_numpy(domain_b_x.astype(np.float32))
                    y_subset_b = torch.from_numpy(domain_b_y.astype(np.int64))
                    x_subset_b, y_subset_b = Variable(x_subset_b, requires_grad=False).cuda(), \
                                 Variable(y_subset_b, requires_grad=False).long().cuda()

                    feat_b_old = temp_old_feature_extractor_network(x_subset_b).detach()
                    feat_b_new = temp_new_feature_extractor_network(x_subset_b)

                    cls_b_old = self.phi(feat_b_old)
                    cls_b_new = self.phi(feat_b_new)

                    loss_main_old = self.ce_loss(cls_b_old, y_subset_b)
                    loss_main_new = self.ce_loss(cls_b_new, y_subset_b)
                    reward = loss_main_old - loss_main_new
                    # calculate the updating rule of omega, here is the max function of h.
                    utility = torch.tanh(reward)
                    # so, here is the min value transfering to the backpropogation.
                    loss_held_out =- utility.sum()
                    meta_loss_held_out += loss_held_out*self.heldout_p

                self.opt_theta.step()
                self.opt_phi.step()

                self.opt_omega.zero_grad()
                meta_loss_held_out.backward()
                self.opt_omega.step()
                torch.cuda.empty_cache()

                print('episode %d' % (_), meta_train_loss_main.data.cpu().numpy(),
                          meta_train_loss_dg.data.cpu().numpy(),
                          meta_loss_held_out.data.cpu().numpy(),
                          )

            if _ % 50 == 0 and _ != 0:
                time_end = datetime.datetime.now()
                epoch = (flags.iteration_size - int(_))%500
                time_cost = epoch * (time_end - time_start).seconds / 60
                time_start = time_end
                torch.cuda.empty_cache()
                print('the number of iteration %d, and it is expected to take another %d minutes to complete..' % (_, time_cost))
                torch.cuda.empty_cache()
                self.validate_workflow(self.batImageGenVals, flags, _)
                torch.cuda.empty_cache()

    def pre_train(self):
        model_path = os.path.join('model_output/PACS/baseline/', 'best_model.tar')
        if os.path.exists(model_path):
            self.load_state_dict(state_dict=model_path)

    def reinit_network_P(self,flags):
        self.beta =  flags.beta
        self.alpha = flags.lr
        self.eta = flags.lr
        self.omega_para = flags.omega
        self.heldout_p = flags.heldout_p
        self.opt_omega = torch.optim.SGD(self.omega.parameters(), lr=self.omega_para, weight_decay=self.weight_decay, momentum=0.9)
