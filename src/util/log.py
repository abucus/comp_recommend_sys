import logging
import os.path as op

log_dir = op.join(op.realpath(__file__),'..','..','..','output')
handlers = {
	'CONSOLE':dict(),
	'FILE':dict()
}

def getLogHandler(level,log_type):

	level = level.upper()
	log_type = log_type.upper()

	assert level in ['DEBUG','INFO','ERROR']
	assert log_type in ['CONSOLE','FILE']

	if level in handlers[log_type]:
		return handlers[log_type][level]
	else:
		handler = None
		if log_type == 'CONSOLE':
			handler = logging.StreamHandler()
		else:
			handler = logging.FileHandler(op.join(log_dir, '%s.log'%level))
		handler.setLevel(getattr(logging, level))
		handlers[log_type][level] = handler
		return handler


def getLogger(name='root',level='DEBUG'):
	logger = logging.getLogger(name)
	logger.setLevel(getattr(logging, level))
	return logger