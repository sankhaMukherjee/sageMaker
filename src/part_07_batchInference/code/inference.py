import numpy as np
import json 

def _return_error(code, message):
    raise ValueError("Error: {}, {}".format(str(code), message))

def input_handler(data, context):


    try:
        print(context)

        # This is expected to be a np array of shape (28,28,1)
        info = np.load( data.read() )
        print(info.shape)
        info = info.reshape((-1, 28, 28, 1))
        
        return json.dumps({ 'instances': info.to_list() })
    except Exception as e:
        _return_error( 01, f'\nError while encoding data\n{data}\nwith context [{}]:\n{e}\n' )

def output_handler(data, context):

    try:
        print(context)
        if data.status_code !=200:
            raise Exception(data.content.decode("utf-8"))

        response_content_type = context.accept_header
        prediction = data.content
        return prediction, response_content_type

    except Exception as e:
        _return_error( 01, f'\nError while encoding data\n{data}\nwith context [{}]:\n{e}\n' )

    return


