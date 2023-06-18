#import pickle as pkl
##合并用函数，输入body和head的文件名、写入文件的格式、写入文件的名称，输出合并后预览，以及自动写入指定文件

def mergee_csv(body_filename='train_bodies.csv',head_filename='train_stances.csv',save_as='csv',save_name='train.csv'):
    #根据给定文件名读取body和head
    _body = pd.read_csv(body_filename)
    _head = pd.read_csv(head_filename)
    #重新标注
    _body=_body.rename(columns={'sentences':'sentences_body','embeddings':'embeddings_body'})
    _head=_head.rename(columns={'sentences':'sentences_head','embeddings':'embeddings_head'})
    merged = _head.merge(_body_tr,on='Body ID',how='inner')
    #预览合并后df
    print('合并后预览：',merged.head(5))
    #写入csv或pkl
    if save_as=='csv':
        merged.to_csv(save_name,index=False)
    elif save_as=='pkl':
        with open(save_name,'wb') as f:
            pkl.dump(merged,f)

def mergee_pkl(body_filename='body_train.pkl',head_filename='stance_train.pkl',save_as='csv',save_name='train.csv'):
    #根据给定文件名读取body和head
    _body = pkl.load(open(body_filename, 'rb'))
    _head = pkl.load(open(head_filename, 'rb'))
    #重新标注
    _body=_body.rename(columns={'sentences':'sentences_body','embeddings':'embeddings_body'})
    _head=_head.rename(columns={'sentences':'sentences_head','embeddings':'embeddings_head'})
    merged = _head.merge(_body_tr,on='Body ID',how='inner')
    #预览合并后df
    print('合并后预览：',merged.head(5))
    #写入csv或pkl
    if save_as=='csv':
        merged.to_csv(save_name,index=False)
    elif save_as=='pkl':
        with open(save_name,'wb') as f:
            pkl.dump(merged,f)            
            
##示例：如果文件名为‘body_test.pkl’和'head_test.pkl',希望保存为名为‘test.pkl’的pkl文件，则如下调用pkl函数
#mergee_csv(‘body_test.pkl’,'head_test.pkl','pkl','test.pkl')
            
