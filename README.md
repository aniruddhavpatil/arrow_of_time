# arrow_of_time
Research Paper Link: https://www.robots.ox.ac.uk/~vgg/publications/2014/Pickup14/pickup14.pdf


# algorithm for the current approach
def kfold_run(path):

    # loop through the given dataset

        # build_dict_of_flow_words(path)

        # for each video in train_dataset

            # get_A_hist_of_video, store it in hist_A, lly generate label_A
            # get_B_hist_of_video, store it in hist_B, lly generate label_A
            # get_C_hist_of_video, store it in hist_C, lly generate label_A
            # get_D_hist_of_video, store it in hist_D, lly generate label_A

        # save the dataset
        # perform pca on each of the hists individually

        # train the dataset on MLP, to give A,B,C,D values given an input of all hist combined together

        # repeat the above steps with the test data except, instead of training on the model predict

    # reset the weights, vocab
