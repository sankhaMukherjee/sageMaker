# import numpy as np
import json, io
import logging 

def _return_error(code, message):
    raise ValueError("Error: {}, {}".format(str(code), message))

def input_handler(data, context):

    logger = logging.getLogger()

    if context.request_content_type == "application/json":

        result = data.read().decode('utf-8')
        result = json.loads( result )
        result = json.dumps( {"instances" : [result] } )
        return result

    else:
        _return_error(
            415, 'Unsupported content type "{}"'.format(context.request_content_type or "Unknown")
        )

def output_handler(data, context):

    try:
        # print(context)
        if data.status_code !=200:
            logger.error('Some other strange error ...')
            # raise Exception(data.content.decode("utf-8"))

        response_content_type = context.accept_header
        prediction            = data.content
        return prediction, response_content_type

    except Exception as e:
        # _return_error( 01, f'\nError while encoding data\n{data}\nwith context [{context}]:\n{e}\n' )
        _return_error( 1, 'Some error' )

    return


