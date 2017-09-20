import json
import numpy as np
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Flowable, SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

from read_data import read_data


def get_feed_dict(config, batch_idxs, dataset):
    feed_dict = {}
    x = []
    q = []
    y1 = []
    y2 = []
    label_smoothing = config['train']['label_smoothing']

    def word2id(word):  # to convert a word to its respective id
        return word  # unknown word

    # TODO: Add characters
    # convert every word to its respective id
    for i in batch_idxs:
        qi = dataset['data']['q'][i]
        rxi = dataset['data']['*x'][i]
        yi = dataset['data']['y'][i]
        xi = dataset['shared']['x'][rxi[0]][rxi[1]]
        q.append(qi)
        x.append(xi)
        # Get all the first indices in the sequence
        y1.append([y[0] for y in yi])
        # Get all the second indices... and correct for -1
        y2.append([y[1]-1 for y in yi])

    answer = [y1, y2]

    return x, q, y1, y2


class MCLine(Flowable):
    """
    Line flowable --- draws a line in a flowable
    http://two.pairlist.net/pipermail/reportlab-users/2005-February/003695.html
    """
    def __init__(self, width, height=0):
        Flowable.__init__(self)
        self.width = width
        self.height = height

    def __repr__(self):
        return "Line(w=%s)" % self.width

    def draw(self):
        """
        draw the line
        """
        self.canv.line(0, self.height, self.width, self.height)


def create_pdf(config, ids=None, Start_Index=None, End_Index=None):

    # Create a reportlab doc
    doc = SimpleDocTemplate("./plots/answers.pdf", pagesize=A4,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    # Define the style
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))

    # Read dev file
    data_dev = read_data(config, 'dev', ref=False, data_filter=True)
    if ids is None:
        ids = data_dev['valid_idxs']
    x, q, y1, y2 = get_feed_dict(config, ids, data_dev)
    story = []
    line = MCLine(450)

    for i in range(len(q)):
        story.append(Paragraph('P: ' + ' '.join(x[i]), styles["Justify"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph('Q: ' + ' '.join(q[i]), styles["Justify"]))
        story.append(Spacer(1, 12))
        for j in range(len(y1[i])):
            story.append(Paragraph('A{}: '.format(j) + ' '.join(x[i][int(y1[i][j]): int(y2[i][j]) + 1]), styles["Justify"]))
            story.append(Spacer(1, 12))
        story.append(Paragraph('FABIR: ' + ' '.join(x[i][int(Start_Index[i]): int(End_Index[i]) + 1]), styles["Justify"]))
        story.append(Spacer(1, 12))
        story.append(line)
        story.append(Spacer(1, 12))
    doc.build(story)


if __name__ == '__main__':
    with open('config.json') as json_data_file:
        config = json.load(json_data_file)
    create_pdf(config)
