import os

def displayNotification(message,title=None):
	"""
		Display an OSX notification with message title an subtitle
		sounds are located in /System/Library/Sounds or ~/Library/Sounds
	"""
	title_part = ''
	if title:
		title_part = 'with title "{0}"'.format(title)
	icon_button = 'with icon caution buttons {\"OK\"}'
	appleScriptNotification = 'display dialog "{0}" {1} {2} '.format(message,title_part,icon_button)
	os.system("osascript -e '{0}'".format(appleScriptNotification))

# displayNotification(message="message", title="Your Patient Needs Your Attention")