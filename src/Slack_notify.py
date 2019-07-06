import slackweb


def post(text="かみまみた"):
    with open("../") as f:
        url = f.read()

    slack = slackweb.Slack(url=url)
    slack.notify(text)
