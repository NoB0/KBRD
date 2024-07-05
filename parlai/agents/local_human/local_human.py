#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Agent does gets the local keyboard input in the act() function.

Example: parlai eval_model -m local_human -t babi:Task1k:1 -dt valid
"""

from parlai.core.agents import Agent
from parlai.core.params import ParlaiParser
from parlai.core.utils import display_messages, load_cands


class LocalHumanAgent(Agent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt=None
    ) -> ParlaiParser:
        """
        Add command-line arguments specifically for this agent.
        """
        agent = parser.add_argument_group('Local Human Arguments')
        agent.add_argument(
            '-fixedCands',
            '--local-human-candidates-file',
            default=None,
            type=str,
            help='File of label_candidates to send to other agent',
        )
        agent.add_argument(
            '--single_turn',
            type='bool',
            default=False,
            help='If on, assumes single turn episodes.',
        )
        return parser

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'localHuman'
        self.episodeDone = False
        self.finished = False
        self.fixedCands_txt = load_cands(self.opt.get('local_human_candidates_file'))
        print(
            "\033[0;90mEnter [DONE] if you want to end the episode, [EXIT] to quit.",
            'highlight\033[0;0m',
        )

    def epoch_done(self):
        return self.finished

    def observe(self, msg):
        print(
            display_messages(
                [msg],
                prettify=self.opt.get('display_prettify', False),
            )
        )

    def act(self):
        reply = dict()
        reply['id'] = self.getID()
        try:
            reply_text = input("\033[0;34mEnter Your Message:\033[0;0m" + ' ')
        except EOFError:
            self.finished = True
            return {'episode_done': True}

        reply_text = reply_text.replace('\\n', '\n')
        reply['episode_done'] = False
        if self.opt.get('single_turn', False):
            reply.force_set('episode_done', True)
        reply['label_candidates'] = self.fixedCands_txt
        if '[DONE]' in reply_text:
            # let interactive know we're resetting
            raise StopIteration
        reply['text'] = reply_text
        if '[EXIT]' in reply_text:
            self.finished = True
            raise StopIteration
        return reply

    def episode_done(self):
        return self.episodeDone
