import os
# https://code-maven.com/display-notification-from-the-mac-command-line
# Add audio feedback?
def displayNotification(message,title=None):
	"""
		Display an OSX notification with message title an subtitle
		sounds are located in /System/Library/Sounds or ~/Library/Sounds
	"""
	title_part = ''
	if title:
		title_part = 'with title "{0}"'.format(title)
	# icon_button = 'with icon caution buttons {\"OK\"}'
	appleScriptNotification = 'display notification "{0}" {1} say "Please listen to the other person"'.format(message,title_part)
	os.system("osascript -e '{0}'".format(appleScriptNotification))

# displayNotification(message="message", title="Your Patient Needs Your Attention")