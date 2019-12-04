



def cross_validate(split_size,train_all_feature,train_all_Y,test_feature,test_Y,outter_loop,G_pool,C_pool):
    results = []
    #train_idx,val_idx = Kfold(len(train_all_feature),split_size)
    train_idx,val_idx = Kfold(len(train_all_feature[0]),split_size)
    prediction = []
    
    #train_all_feature = np.reshape(train_all_feature,(len(train_all_feature),len(train_all_feature[0])))
    #train_all_Y = np.reshape(train_all_Y,(len(train_all_Y),len(train_all_Y[0])))
    
    #test_feature = np.reshape(test_feature,(len(test_feature),len(test_feature[0])))
    
    
    for k in range(split_size):
        train_feature0 = [train_all_feature[0][i] for i in train_idx[k]]
        train_feature1 = [train_all_feature[1][i] for i in train_idx[k]]
        train_feature2 = [train_all_feature[2][i] for i in train_idx[k]]
        
        train_feature = []
        val_feature = []
        
        train_feature.append(train_feature0)
        train_feature.append(train_feature1)
        train_feature.append(train_feature2)
        
        train_Y = [train_all_Y[i] for i in train_idx[k]]
        
        val_feature0 = [train_all_feature[0][i] for i in val_idx[k]]
        val_feature1 = [train_all_feature[1][i] for i in val_idx[k]]
        val_feature2 = [train_all_feature[2][i] for i in val_idx[k]]
        
        val_Y = [train_all_Y[i] for i in val_idx[k]]
        val_feature.append(val_feature0)
        val_feature.append(val_feature1)
        val_feature.append(val_feature2)
        
        #train_feature = np.reshape(train_feature,(len(train_feature),len(train_feature[0])))
        #val_feature = np.reshape(val_feature,(len(val_feature),len(val_feature[0])))
        
    
        print('outter_loop',outter_loop)
        print('inner_loop',k,'start')
        
        #epoch_result = []
        #for epoch in epoch_list:
        print('start best para search')
        #run_train(session, train_feature, train_Y, epoch)
        #epoch_result.append(session.run(accuracy, feed_dict={fea: val_feature, clas: val_Y}))
        #best_regu,best_W,best_epoch = run_train(train_feature,train_Y,starting_epoch,regularizer_pool,W_pool,val_feature,val_Y,learning_rate)
        test_score,preds,best_c,best_g,best_order = run_train(train_feature,train_Y,G_pool,C_pool,val_feature,val_Y,test_feature,test_Y)
        #print('test score is', test_score)
        #print('start best para training/test')
        #score,preds = run_test(train_all_feature,train_all_Y,test_feature,test_Y,best_regu,best_W,best_epoch,learning_rate)
        #print('finished best epoch training')
        results.append(test_score)
        prediction.append(preds)
        
        #print(preds)
        print('best c is',best_c)
        print('best g is',best_g)
        print('best oder is',best_order)
        print('this run accuracy is', results[-1])
        print('inner_loop',k,'ends')
    #print(prediction)
    #prediction = np.reshape(9,len(prediction[0]))
    
    prediction = np.array(prediction)
    #print(prediction.shape)
    pre = []
    for i in range(prediction.shape[1]):
        pre.append(Counter(prediction[:,i]).most_common(1)[0][0])
    test_acc = np.mean(np.equal(pre,test_Y))
    print(pre)
    return (results,test_acc)

#def run_test(train_all_feature,train_all_Y,test_feature,test_Y,best_regu,best_W,best_epoch,learning_rate):
#    model = geometric_scattering_classifier(best_W,best_regu,learning_rate)
#    model.fit(train_all_feature,train_all_Y,epochs=best_epoch,batch_size=64,shuffle=False)
#    loss,score = model.evaluate(test_feature,test_Y)
#    preds = model.predict(test_feature)
#    return (score,preds)


def run_train(train_feature,train_Y,G_pool,C_pool,val_feature,val_Y,test_feature,test_Y):
    temp = 0
    for c in C_pool:
        for g in G_pool:
            for i in range(len(train_feature)):
                model = SVC(kernel='rbf',C=c,gamma=g)
                #model = SVC(kernel='rbf',C=c)
                model.fit(train_feature[i],train_Y)
                score = model.score(val_feature[i],val_Y)
                print('training score is',model.score(train_feature[i],train_Y))
                print('validation score is',score)
                if score >temp:
                    temp =score
                    best_order = i
                    test_score = model.score(test_feature[i],test_Y)
                    preds = model.predict(test_feature[i])
                    best_c = c
                    best_g = g
    return (test_score,preds,best_c,best_g,best_order)
#def write_to_file(index):
    #with open('shuffle_index_0.txt','w') as file:
        #for i in index:
            #file.write(str(i))
            #file.write('\t')
            
            
def Kfold(length,fold):
    size = np.arange(length).tolist()
    train_index = []
    val_index = []
    rest = length % fold
    fold_size = int(length/fold)
    temp_fold_size = fold_size
    for i in range(fold):
        temp_train = []
        temp_val = []
        if rest>0:
            temp_fold_size = fold_size+1
            rest = rest -1
            temp_val = size[i*temp_fold_size:+i*temp_fold_size+temp_fold_size]
            temp_train = size[0:i*temp_fold_size] + size[i*temp_fold_size+temp_fold_size:]
        else:
            temp_val = size[(length % fold)*temp_fold_size+(i-(length % fold))*fold_size
                            :(length % fold)*temp_fold_size+(i-(length % fold))*fold_size+fold_size]
            temp_train = size[0:(length % fold)*temp_fold_size+(i-(length % fold))*fold_size] + size[(length % fold)*temp_fold_size+(i-(length % fold))*fold_size+fold_size:]
        train_index.append(temp_train)
        val_index.append(temp_val)
    return (train_index,val_index)
