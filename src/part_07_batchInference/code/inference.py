import numpy as np
import json, io
from logging 

def _return_error(code, message):
    raise ValueError("Error: {}, {}".format(str(code), message))

def input_handler(data, context):

    logger = logging.getLogger()

    logger.warning('------------- [warning]---------------------')
    logger.debug('--------------- [Debug] ----------------------')
    logger.error('---------------- [error] ---------------------')

    if context.request_content_type == "application/json":

        # image = np.load(data)
        # image = np.expand_dims(image, axis=0)
        # instance = [{"dt_float": image.tolist()}]
        # result = json.dumps( { "instances" : instance } )
        # result = '{ "instances" : [[0]]}'
        result = data.read().decode('utf-8')
        result = json.loads( result )
        result = np.array(result).astype(np.float32)
        result = result.reshape(28, 28)
        # result = np.expand_dims(result, axis=0)
        # result = json.dumps( {"instances" : [result] } )
        result = json.dumps( result )
        return None

    else:
        _return_error(
            415, 'Unsupported content type "{}"'.format(context.request_content_type or "Unknown")
        )

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


