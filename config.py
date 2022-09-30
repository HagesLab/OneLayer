import logging
import logging.config
import os

def init_logging(name=__name__):
    logger = logging.getLogger(name)

    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'format': '%(asctime)s %(levelname)s %(name)s %(message)s',
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'default',
                'level': os.getenv('LOG_LEVEL', 'DEBUG'),
            },
        },
        'loggers': {
            '': {
                'handlers': ['console'],
                'level': 'DEBUG',
            },
        },
    })

    return logger
