import re
import sys
import matplotlib as mpl
import tensorflow as tf
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def get_word_span(context, spans, start, stop):
	search_index = 0
	word_id = []
	word_index = []
	for word_idx, word in enumerate(spans):
		word_init = context.find(word, search_index)
		assert word_init >= 0
		word_end = word_init + len(word)
		if (stop > word_init and start < word_end):
			word_id.append(word_idx)
			word_index.append([word_init, word_end])
		search_index = word_end

	assert len(word_id) > 0, "{} {} {} {}".format(context, spans, start, stop)
	return word_id[0], (word_id[-1] + 1), word_index[0][0], word_index[-1][0]

def get_words_pos(context, words):
    search_index = 0
    word_pos =[]
    for word in words:
        word_init = context.find(word, search_index)
        assert word_init >= 0
        word_end = word_init + len(word)-1
        word_pos.append([word_init,word_end])
        search_index = word_end+1
    return word_pos


def process_tokens(temp_tokens):
	tokens = []
	for token in temp_tokens:
		flag = False
		l = ("-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
		# \u2013 is en-dash. Used for number to nubmer
		# l = ("-", "\u2212", "\u2014", "\u2013")
		# l = ("\u2013",)
		tokens.extend(re.split("([{}])".format("".join(l)), token))
		tokens = list(filter(None, tokens))
	return tokens


def send_mail(attach_dir, subject,body):
    COMMASPACE = ', '
    sender = 'jorgematlab93@gmail.com'
    gmail_password = 'AmeMatlab12.'
    recipients = ['jorge.silva93@gmail.com', 'alvaro.hc.correia@gmail.com']

    # Create the enclosing (outer) message
    outer = MIMEMultipart()
    outer['Subject'] = subject
    outer['To'] = COMMASPACE.join(recipients)
    outer['From'] = sender
    outer.preamble = 'You will not see this in a MIME-aware mail reader.\n'
    outer.attach(MIMEText(body, 'plain'))
    # List of attachments
    attachments = attach_dir

    # Add the attachments to the message
    for file in attachments:
        try:
            with open(file, 'rb') as fp:
                msg = MIMEBase('application', "octet-stream")
                msg.set_payload(fp.read())
            encoders.encode_base64(msg)
            msg.add_header('Content-Disposition', 'attachment', filename=os.path.basename(file))
            outer.attach(msg)
        except:
            print("Unable to open one of the attachments. Error: ", sys.exc_info()[0])

    composed = outer.as_string()

    # Send the email
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as s:
            s.ehlo()
            s.starttls()
            s.ehlo()
            s.login(sender, gmail_password)
            s.sendmail(sender, recipients, composed)
            s.close()
        print("Email sent!")
    except:
        print("Unable to send the email. Error: ", sys.exc_info()[0])


def plot_q_type(X, N, directory):
    labels = ['what', 'which', 'who', 'hmany', 'hmuch', 'hlong', 'how', 'when', 'where', 'why', 'other']
    size_X = len(X)

    ind = np.arange(size_X)  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, X, width, color='g')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('F1-Score (%)')
    ax.set_title('F1-Score against Question Type')
    ax.set_xticks(ind)
    ax.set_xticklabels(labels, rotation='vertical')

    ax.legend(rects1, ['FABIR'])

    def autolabel(rects,N):
        """
        Attach a text label above each bar displaying its height
        """
        i = 0
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%d' % int(N[i]),
                    ha='center', va='bottom')
            i+=1

    autolabel(rects1,N)
    plt.savefig(directory)

def plot_line(Y, plot_type, directory):
    fig, ax = plt.subplots()
    size_Y = len(Y)
    ind = np.arange(size_Y)
    if plot_type == 'par_len':
        xlabel='Passage Length'
        labels = ['60','100','140','180','220', '260','>260']
    elif plot_type == 'q_len':
        xlabel='Question Length'
        labels =  ['7','11','15','19','>19']
    elif plot_type == 'ans_len':
        xlabel='Answer Length'
        labels =  ['1','2','3','4','5','6','7','8','9','10', '>10']
    plt.plot(ind,Y, 'go', ind,Y, 'g')
    ax.set_xticks(ind)
    ax.set_xticklabels(labels, rotation='vertical')
    ax.set_ylabel('F1-Score (%)')
    ax.set_xlabel(xlabel)
    ax.set_title('F1-Score against '+xlabel)
    plt.grid()
    plt.savefig(directory)

def plot(X, EM, F1, save_dir):
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(X, EM[0], label='train')
    axarr[0].plot(X, EM[1], label='dev')
    axarr[0].set_title('EM (%)')
    axarr[0].grid(True)
    axarr[1].plot(X, F1[0], label='train')
    axarr[1].plot(X, F1[1], label='dev')
    axarr[1].set_title('F1 (%)')
    axarr[1].grid(True)

    handles, labels = axarr[0].get_legend_handles_labels()
    axarr[0].legend(handles[::-1], labels[::-1])
    handles, labels = axarr[1].get_legend_handles_labels()
    axarr[1].legend(handles[::-1], labels[::-1])
    plt.savefig(save_dir)


def F1(y1, y2, prob_yp1, prob_yp2):
    y1 = tf.cast(tf.argmax(y1, axis=-1), tf.float32)
    y2 = tf.cast(tf.argmax(y2, axis=-1), tf.float32)
    yp1 = tf.cast(tf.argmax(prob_yp1, axis=-1), tf.float32)
    yp2 = tf.cast(tf.argmax(prob_yp2, axis=-1), tf.float32)
    TT = tf.minimum(tf.cast(y2 + 1, tf.float32), tf.cast(yp2 + 1, tf.float32)) - tf.maximum(y1, yp1)
    TT = tf.maximum(TT, 0)
    FT = yp2 + 1 - yp1 - TT
    FF = y2 + 1 - y1 - TT
    a = TT/(TT+FT)
    b = TT/(TT+FF)
    ab = tf.multiply(a, b)
    F1 = tf.where(tf.equal(ab, 0.0), tf.zeros_like(y1), (2.0/(1.0/a+1.0/b)))

    return F1


def get_answer(Start_Index,End_Index,batch_idxs, data):
    answer = []
    dictionary = {}
    data['data'].keys()
    rx = data['data']['*x']
    x = data['shared']['x']
    p = data['shared']['p']
    cy = data['data']['cy']
    ids = data['data']['ids']
    word_pos = data['data']['word_pos']
    for i in range(len(Start_Index)):
        addr_1, addr_2 = rx[batch_idxs[i]][0], rx[batch_idxs[i]][1]
        ID = ids[batch_idxs[i]]
        char_init = word_pos[addr_1][addr_2][Start_Index[i]][0]
        char_end = word_pos[addr_1][addr_2][End_Index[i]][1]
        answer = p[addr_1][addr_2][char_init:char_end+1]
        dictionary = {**dictionary, **{ID: answer}}
    return dictionary

def maxSubArraySum(a):
    Start_Index, End_Index, Prob = [], [], []
    for i in range(len(a)):
        max_so_far = -1e5
        max_ending_here = 0.0
        ind_so_far = [0,0]
        ind_here = [0,0]
        for j in range(len(a[i])):
            max_ending_here = max_ending_here + a[i][j]
            ind_here[1] = j
            if (max_so_far < max_ending_here):
                max_so_far = max_ending_here
                ind_so_far[0] = ind_here[0]
                ind_so_far[1] = ind_here[1]
            if max_ending_here < 0.0:
                ind_here[0] = j+1
                max_ending_here = 0.0
        Start_Index.append(ind_so_far[0])
        End_Index.append(ind_so_far[1])
        Prob.append(max_so_far)
    return Start_Index, End_Index, Prob

def EM_and_F1(answer, answer_est):
    EM = []
    F1 = []
    y1_correct = []
    y2_correct = []
    y2_greater_y1_correct = []
    y1_est, y2_est = answer_est
    y1, y2 = answer
    for i in range(len(y1_est)):
        y1_correct_i = []
        y2_correct_i = []
        EM_i = []
        F1_i = []
        for j in range(len(y1[i])):
            y1_correct_i.append(1.0 if y1[i][j] == y1_est[i] else 0.0)
            y2_correct_i.append(1.0 if y2[i][j] == y2_est[i] else 0.0)
            EM_i.append(1.0 if y1[i][j] == y1_est[i] and y2[i][j] == y2_est[i] else 0.0)
            TT = max([min([y2[i][j]+1, y2_est[i]+1]) - max([y1[i][j], y1_est[i]]), 0])
            FT = y2_est[i]+1-y1_est[i]-TT
            FF = y2[i][j]+1-y1[i][j]-TT
            a = TT/(TT+FT)
            b = TT/(TT+FF)
            F1_i.append(2/(1/a+1/b) if a != 0 and b != 0 else 0)
        y2_greater_y1_correct.append(1.0 if y2_est[i] >= y1_est[i] else 0.0)
        y1_correct.append(max(y1_correct_i))
        y2_correct.append(max(y2_correct_i))
        EM.append(max(EM_i))
        F1.append(max(F1_i))
    return [100*sum(EM)/len(EM), 100*sum(F1)/len(F1), sum(y1_correct)/len(y1_correct), sum(y2_correct)/len(y2_correct), sum(y2_greater_y1_correct)/len(y2_greater_y1_correct)]
