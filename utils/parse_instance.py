import os
import re
import itertools
import random
from math import exp
import numpy as np
from pprint import pprint


class InstanceParser(object):
    def __init__(self, domain, instance):
        self.domain = domain
        self.instance = instance
        curr_dir_path = os.path.dirname(os.path.realpath(__file__))
        self.domain_file = os.path.abspath(
            os.path.join(curr_dir_path,
                         "../rddl/domains/{}_mdp.rddl".format(self.domain)))
        self.instance_file = os.path.abspath(
            os.path.join(
                curr_dir_path, "../rddl/domains/{}_inst_mdp__{}.rddl".format(
                    self.domain, self.instance.replace('.', '_'))))
        self.parsed_instance_file = os.path.abspath(
            os.path.join(
                curr_dir_path, "../rddl/parsed/{}_inst_mdp__{}".format(
                    self.domain, self.instance.replace('.', '_'))))

        self.dot_file = os.path.abspath(
            os.path.join(
                curr_dir_path, "../rddl/dbn/{}_inst_mdp__{}.dot".format(
                    self.domain, self.instance.replace('.', '_'))))

        try:
            with open(self.domain_file) as f:
                domain_file_str = f.readlines()
        except UnicodeDecodeError as e:
            with open(self.domain_file, encoding="ISO-8859-1") as f:
                domain_file_str = f.readlines()

        with open(self.instance_file) as f:
            instance_file_str = f.read()

        with open(self.parsed_instance_file) as f:
            parsed_instance_file_str = f.read()

        with open(self.dot_file) as f:
            dot_instance_file_str = f.read()

        # parameters from dbn
        self.color = {
            'initial_state': 'lightblue',
            'final_state': 'gold1',
            'action': 'olivedrab1'
        }

        self.object_names = set()  # contains name of nodes form dbn

        self.para_state_names = set()  # contains name of state templates
        self.para_action_names = set()  # contains name of action templates

        self.state_object_names = set(
        )  # contains name of nodes(states) form dbn
        self.para_state_of_objects = {
        }  # contains fluent state template for an object

        self.action_object_names = set(
        )  # contains name of nodes(actions) form dbn
        self.para_action_of_objects = {
        }  # contains action template for an object

        self.para_state_connections = set(
        )  # contains tuple of (state object,state object) connections in objects
        self.para_action_connections = set(
        )  # contains tuple of (state object,action object) connections in objects

        self.unpara_state_names = set(
        )  # contains name of state templates (unparameterized)
        self.unpara_action_names = set(
        )  # contains name of action templates (unparameterized)

        self.num_actions_dict = {
        }  # contains names of action template for a node in action

        self.parse_dot_file(dot_instance_file_str)

        print('all objects')
        pprint(self.object_names)

        print('state templates')
        pprint(self.para_state_names)

        print('action templates')
        pprint(self.para_action_names)

        print('state objects')
        pprint(self.state_object_names)

        print('state mappings from object')
        pprint(self.para_state_of_objects)

        print('action objects')
        pprint(self.action_object_names)

        print('action mappings from object')
        pprint(self.para_action_of_objects)

        print('state to state connections')
        pprint(self.para_state_connections)

        print('state to action connection')
        pprint(self.para_action_connections)

        print('unpara state templates')
        pprint(self.unpara_state_names)

        print('unpara action templates')
        pprint(self.unpara_action_names)

        # parameters from domain files

        self.zero_nf_names = []
        self.unary_nf_names = []
        self.multiple_nf_names = []

        self.objects_in_domain_file = []  # types of objects in domain file
        self.object_name_to_type = {}  # object -> type
        self.type_of_nodes_in_graph = set()  # types of state nodes

        # contains vector of nf values for nodes
        self.para_state_of_objects_nf_values = {}
        self.para_state_of_objects_nf_values_names = {}

        # contains vector of f values for nodes
        self.para_state_of_objects_f_values = {}
        self.para_state_of_objects_f_values_names = {}

        for k in self.para_state_of_objects.keys():
            self.para_state_of_objects_nf_values[k] = []
            self.para_state_of_objects_f_values[k] = []

            self.para_state_of_objects_nf_values_names[k] = []
            self.para_state_of_objects_f_values_names[k] = []

        for line in domain_file_str:
            line = line.strip()
            line = line[:line.find('//')].strip()
            if 'object' in line:
                self.objects_in_domain_file.append(
                    line[:line.find(':')].strip())
            elif 'non-fluent' in line:

                values = line[line.find('{') + 1:line.find('}')].split(',')
                name = line[:line.find(':')].strip()

                default = values[-1][values[-1].find("=") + 1:].strip()
                type_val = None
                if values[-2].strip() == 'real':
                    default = float(default)
                    type_val = float
                elif values[-2].strip() == 'bool':
                    if default == 'false':
                        default = 0
                    else:
                        default = 1
                    type_val = bool

                if name.count(',') >= 1:
                    parameters = name[name.find('(') +
                                      1:name.find(')')].strip().replace(
                                          ' ', '')
                    self.multiple_nf_names.append(
                        (name[:name.find('(')].strip(), type_val, default,
                         parameters))
                elif name.count('(') >= 1:
                    parameters = name[name.find('(') +
                                      1:name.find(')')].strip().replace(
                                          ' ', '')
                    self.unary_nf_names.append((name[:name.find('(')].strip(),
                                                type_val, default, parameters))
                else:
                    self.zero_nf_names.append((name.strip(), type_val, default,
                                               None))

            elif 'action-fluent' in line:
                pass

        for ob_type in self.objects_in_domain_file:
            c = re.findall('{0} : {{.*?}}'.format(ob_type),
                           instance_file_str)[0]
            temp = c[c.find("{") + 1:c.find("}")].split(',')
            temp = [i.strip() for i in temp]

            for ob_name in temp:
                self.object_name_to_type[ob_name] = ob_type

        print('object to type')
        pprint(self.object_name_to_type)

        for ob_name in self.object_names:
            temp = ob_name.split(',')
            temp = [self.object_name_to_type[i] for i in temp]
            self.type_of_nodes_in_graph.add(','.join(temp))

        print('type of nodes in graph')
        pprint(self.type_of_nodes_in_graph)

        for k, nf in enumerate(sorted(self.unary_nf_names)):

            temp_flag = False
            for key in self.type_of_nodes_in_graph:
                if nf[-1] in key:
                    temp_flag = True

            if not temp_flag:
                continue

            if nf[1] is not bool:
                continue

            pr = re.findall('{}\(.*?;'.format(nf[0]), instance_file_str)

            for key in self.para_state_of_objects_nf_values.keys():
                self.para_state_of_objects_nf_values[key].append(nf[2])
            for key in self.para_state_of_objects_nf_values_names.keys():
                self.para_state_of_objects_nf_values_names[key].append(nf[0])

            for c in pr:
                node = c[c.find("(") + 1:c.find(")")]
                node = node.strip().replace(' ', '')
                for key in self.para_state_of_objects_nf_values.keys():
                    if node in key:
                        if nf[1] == bool:
                            self.para_state_of_objects_nf_values[key][
                                -1] = 1 - nf[2]

                        else:
                            self.para_state_of_objects_nf_values[key][-1] = nf[
                                1](c[c.find('=') + 1:-1])

        for k, nf in enumerate(sorted(self.multiple_nf_names)):

            if nf[-1] not in self.type_of_nodes_in_graph:
                continue

            if nf[1] is not bool:
                continue

            pr = re.findall('{}\(.*?;'.format(nf[0]), instance_file_str)

            for key in self.para_state_of_objects_nf_values.keys():
                self.para_state_of_objects_nf_values[key].append(nf[2])
            for key in self.para_state_of_objects_nf_values_names.keys():
                self.para_state_of_objects_nf_values_names[key].append(nf[0])

            for c in pr:
                node = c[c.find("(") + 1:c.find(")")]
                node = node.strip().replace(' ', '')

                if node in self.para_state_of_objects_nf_values.keys():
                    self.para_state_of_objects_nf_values[node][-1] = 1 - nf[2]

        print('names of nf vector for object')
        pprint(self.para_state_of_objects_nf_values_names)

        print('nf values for object')
        pprint(self.para_state_of_objects_nf_values)
        print()

        # initial state

        psr = parsed_instance_file_str[
            parsed_instance_file_str.find("#####TASK##### Here") +
            len("#####TASK##### Here"):parsed_instance_file_str.find(
                "#####ACTION FLUENTS#####")].strip().split(
                    "## initial state\n")[1]

        psr = psr.strip().split("\n")
        self.initial_state = list(map(int, psr[0].strip().split(' ')))

        # mapping form action to numbers

        self.action_to_num = {}

        action_str = parsed_instance_file_str[
            parsed_instance_file_str.find("#####ACTION FLUENTS#####") +
            len("#####ACTION FLUENTS#####"):parsed_instance_file_str.find(
                "#####DET STATE FLUENTS AND CPFS#####")].strip().split(
                    "## index\n")[1:]

        action_str = [i.strip().split('\n') for i in action_str]

        for ac in action_str:
            self.action_to_num[ac[2].replace(' ', '')] = int(ac[0]) + 1

        print('action to num')
        pprint(self.action_to_num)

        # mapping form state to numbers

        self.state_to_num = {}

        det_str = parsed_instance_file_str[
            parsed_instance_file_str.find(
                "#####DET STATE FLUENTS AND CPFS#####") +
            len("#####DET STATE FLUENTS AND CPFS#####"):
            parsed_instance_file_str.find(
                "#####PROB STATE FLUENTS AND CPFS#####")].strip().split(
                    "## index\n")[1:]

        det_str = [i.strip().split('\n') for i in det_str]

        for ac in det_str:
            self.state_to_num[ac[2].replace(' ', '')] = int(ac[11])

        prob_str = parsed_instance_file_str[
            parsed_instance_file_str.find(
                "#####PROB STATE FLUENTS AND CPFS#####") +
            len("#####PROB STATE FLUENTS AND CPFS#####"):
            parsed_instance_file_str.find("#####REWARD#####")].strip().split(
                "## index\n")[1:]

        prob_str = [i.strip().split('\n') for i in prob_str]

        for ac in prob_str:
            self.state_to_num[ac[2].replace(' ', '')] = int(ac[11])

        print('state to num')
        pprint(self.state_to_num)

        # Making of actual graph for neural network

        self.node_dict = {}

        for k, obj in enumerate(sorted(self.state_object_names)):
            self.node_dict[obj] = k

        print('object to node num')
        pprint(self.node_dict)

        self.adjacency_list = {}
        for a, b in sorted(self.para_state_connections):
            if self.node_dict[a] not in self.adjacency_list:
                self.adjacency_list[self.node_dict[a]] = [self.node_dict[b]]
            else:
                self.adjacency_list[self.node_dict[a]].append(
                    self.node_dict[b])

        print('adjacency list')
        pprint(self.adjacency_list)

        # making of features for network

        self.num_graph_action = 1  #noop
        self.num_parameter_actions = 0

        self.graph_nf_features = []
        self.nf_features = [None] * len(self.state_object_names)

        for key, item in self.para_state_of_objects_nf_values.items():
            self.nf_features[self.node_dict[key]] = item

        print("nf features")
        pprint(self.nf_features)

        self.fluent_state_dict = {}

        for i, sn in enumerate(sorted(self.para_state_names)):
            self.fluent_state_dict[sn] = i

        print("state fluent dict")
        pprint(self.fluent_state_dict)

        self.graph_f_features = []
        self.f_features = [[None] * len(self.para_state_names)
                           for _ in range(len(self.state_object_names))]

        print("initial fluent features")
        pprint(self.get_fluent_features(self.initial_state))

        print("graph fluent features")
        pprint(self.get_graph_fluent_features(self.initial_state))

        try:
            self.fluent_feature_dims = len(self.f_features[0])
        except IndexError as e:
            self.fluent_feature_dims = 0
        try:
            self.nonfluent_feature_dims = len(self.nf_features[0])
        except IndexError as e:
            self.nonfluent_feature_dims = 0

        self.num_graph_action = len(self.unpara_action_names) + 1

        self.action_template_to_num = {}

        for k, tp in enumerate(sorted(self.unpara_action_names)):
            self.action_template_to_num[tp] = k + 1

        for k, tp in enumerate(sorted(self.para_action_names)):
            self.action_template_to_num[tp] = k + self.num_graph_action

        print('action template to num')
        pprint(self.action_template_to_num)

        self.detailed_action = {}

        # parameter action details
        for st, ob, ac in sorted(self.para_action_connections):
            try:
                action_num = self.action_to_num[ac + '(' + ob + ')']
            except KeyError as e:
                continue
            template_num = self.action_template_to_num[ac]
            node_num = self.node_dict[st]

            if action_num not in self.detailed_action.keys():
                self.detailed_action[action_num] = (template_num,
                                                    set([node_num]))
            else:
                self.detailed_action[action_num][1].add(node_num)

        # unparameter action details
        self.detailed_action[0] = (0, set())
        i = 1
        for k in self.action_to_num.keys():
            action_num = self.action_to_num[k]
            if action_num not in self.detailed_action.keys():
                self.detailed_action[action_num] = (i, set())
                i += 1

        print('detailed action: template, [nodes of states]')
        pprint(self.detailed_action)

        self.num_types_action = -1
        for action_template, _ in self.detailed_action.values():
            self.num_types_action = max(self.num_types_action, action_template)
        self.num_types_action += 1

        print('num types action')
        pprint(self.get_num_type_actions())

    def parse_dot_file(self, dot_instance_file_str):
        file_str = dot_instance_file_str.strip().split(';')

        for k, line in enumerate(file_str):
            if self.color['final_state'] in line:
                c = re.findall('\".*?\"', line)[0][1:-1]
                if '$' in c:
                    obj = c[c.find('$'):c.find(')')].replace('$', '').replace(
                        ' ', '')
                    state_var = c[:c.find('\'')].replace('$', '')

                    self.object_names.add(obj)
                    self.para_state_names.add(state_var)

                    if obj not in self.para_state_of_objects:
                        self.para_state_of_objects[obj] = set({state_var})
                    else:
                        self.para_state_of_objects[obj].add(state_var)
                else:
                    self.unpara_state_names.add(c.replace('\'', ''))

            elif self.color['action'] in line:
                # c = re.findall('\".*?\"', line)[0][1:-1]
                c = line[:line.find('[color')].strip().replace('\"', '')
                if '$' in c:
                    obj = c[c.find('$'):c.find(')')].replace('$', '').replace(
                        ' ', '')
                    action_var = c[:c.find('(')].replace('$', '')

                    self.object_names.add(obj)
                    self.para_action_names.add(action_var)

                    if obj not in self.para_action_of_objects:
                        self.para_action_of_objects[obj] = set({action_var})
                    else:
                        self.para_action_of_objects[obj].add(action_var)
                else:
                    self.unpara_action_names.add(c.replace('$', ''))

            elif '->' in line:
                sp = line.split('->')
                f = sp[0].strip()
                t = sp[1].strip()
                if '$' not in f or '$' not in t:
                    continue
                from_var = f[f.find('\"') + 1:f.find('(')].replace('$', '')
                from_obj = f[f.find('(') + 1:f.find(')')].replace('$',
                                                                  '').replace(
                                                                      ' ', '')

                to_var = t[t.find('\"') + 1:t.find('(')].replace('$',
                                                                 '').replace(
                                                                     '\'', '')
                to_obj = t[t.find('(') + 1:t.find(')')].replace('$',
                                                                '').replace(
                                                                    ' ', '')

                if from_var in self.para_state_names:
                    self.para_state_connections.add((from_obj, to_obj))

                elif from_var in self.para_action_names:
                    self.para_action_connections.add((to_obj, from_obj,
                                                      from_var))

                if (from_var, from_obj) not in self.num_actions_dict.keys():
                    self.num_actions_dict[(from_var, from_obj)] = set({to_var})
                else:
                    self.num_actions_dict[(from_var, from_obj)].add(to_var)

        self.state_object_names = set(self.para_state_of_objects.keys())
        self.action_object_names = set(self.para_action_of_objects.keys())

    def get_adjacency_list(self):
        return self.adjacency_list

    def get_fluent_features(self, state):
        for st in self.para_state_names:
            for node in self.state_object_names:
                stn = st + '(' + node + ')'
                try:
                    self.f_features[self.node_dict[node]][
                        self.fluent_state_dict[st]] = state[self.state_to_num[
                            stn]]
                except KeyError as e:
                    self.f_features[self.node_dict[node]][
                        self.fluent_state_dict[st]] = 0
                except:
                    print("something is wrong")
                    exit(0)

        return self.f_features

    def get_graph_fluent_features(self, state):
        gf = []
        for key, item in self.state_to_num.items():
            if key in self.unpara_state_names:
                gf.append(state[item])
        return gf

    def get_feature_dims(self):
        return self.fluent_feature_dims, self.nonfluent_feature_dims

    def get_num_actions(self):
        return len(list(self.detailed_action.keys()))

    def get_action_details(self):
        return self.detailed_action

    def get_nf_features(self):
        return self.nf_features

    def get_stats(self):
        num_action_templates = len(list(
            self.action_template_to_num.keys())) + 1
        action = {}
        for k in self.detailed_action.keys():
            if self.detailed_action[k][0] in action.keys():
                action[self.detailed_action[k][0]].add(
                    len(self.detailed_action[k][1]))
            else:
                action[self.detailed_action[k][0]] = set(
                    {len(self.detailed_action[k][1])})

        return (num_action_templates, action)

    def get_num_nodes(self):
        return len(self.node_dict.keys())

    def get_num_graph_fluents(self):
        return len(list(self.unpara_state_names))

    def get_num_type_actions(self):
        return self.num_types_action


def main():

    stats = {
        'sysadmin': [],
        'wildfire': [],
        'recon': [],
        'game_of_life': [],
        'academic_advising': [],
        'navigation': [],
        'crossing_traffic': [],
        'elevators': [],
        'skill_teaching': [],
        'tamarisk': [],
        'traffic': [],
        'triangle_tireworld': []
    }
    for i in range(2, 3):
        # print("sysadmin_{}".format(i))
        # stats['sysadmin'].append(
        #     InstanceParser('sysadmin', '{}'.format(i)).get_stats())

        # print("wildfire")
        # stats['wildfire'].append(
        #     InstanceParser('wildfire', '{}'.format(i)).get_stats())

        # print("recon")
        # stats['recon'].append(
        #     InstanceParser('recon', '{}'.format(i)).get_stats())

        # print("game_of_life")
        # stats['game_of_life'].append(
        #     InstanceParser('game_of_life', '{}'.format(i)).get_stats())

        # print("academic_advising")
        # stats['academic_advising'].append(
        #     InstanceParser('academic_advising', '{}'.format(i)).get_stats())

        # print("navigation")
        # stats['navigation'].append(
        #     InstanceParser('navigation', '{}'.format(i)).get_stats())

        # print("crossing_traffic")
        # stats['crossing_traffic'].append(
        #     InstanceParser('crossing_traffic', '{}'.format(i)).get_stats())

        # print("elevators")
        # stats['elevators'].append(
        #     InstanceParser('elevators', '{}'.format(i)).get_stats())

        # print("skill_teaching")
        # stats['skill_teaching'].append(
        #     InstanceParser('skill_teaching', '{}'.format(i)).get_stats())

        # print("tamarisk")
        # stats['tamarisk'].append(
        #     InstanceParser('tamarisk', '{}'.format(i)).get_stats())

        # print("traffic")
        # stats['traffic'].append(
        #     InstanceParser('traffic', '{}'.format(i)).get_stats())

        print("triangle_tireworld")
        stats['triangle_tireworld'].append(
            InstanceParser('triangle_tireworld', '{}'.format(i)).get_stats())

    # pprint(stats)


if __name__ == '__main__':
    main()
