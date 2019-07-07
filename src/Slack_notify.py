import slackweb


def post(text="かみまみた"):
    with open("../slack_url.txt") as f:
        url = f.read()

    slack = slackweb.Slack(url=url)
    slack.notify(text=text)
