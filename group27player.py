from pypokerengine.players import BasePokerPlayer
import random as rand
import pprint
import time
import math
import copy
import itertools
from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.engine.card import Card

class Group27Player(BasePokerPlayer):

  cards = ["CA", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "CT", "CJ", "CQ", "CK",
           "SA", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "ST", "SJ", "SQ", "SK",
           "DA", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "DT", "DJ", "DQ", "DK",
           "HA", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "HT", "HJ", "HQ", "HK"]

  class MCTSNode:
    def __init__(self, round_state, hole_card, known_player, current_player, parent, outer_instance):
      self.round_state = round_state
      self.hole_card = hole_card
      self.known_player = known_player
      self.current_player = current_player
      self.parent = parent
      self.actions = outer_instance.next_actions(round_state, hole_card)
      self.children = {}
      self.visited = 0
      self.expected = 0

  # a helper function to return some info about the node such as pot amount
  # last opponent action and last action by us
  def get_info(self, node):
    dict = {}

    pot = node.round_state["pot"]["main"]["amount"]
    pot_factor = (pot + 0.0) / node.round_state["small_blind_amount"]

    if pot_factor <= 10:
      pot_factor = 10
    elif pot_factor <= 20:
      pot_factor = 20
    elif pot_factor <= 30:
      pot_factor = 30
    else:
      pot_factor = 40

    dict["pot"] = pot_factor
    me_uuid = node.round_state["seats"][node.known_player]["uuid"]
    opp_uuid = node.round_state["seats"][1 - node.known_player]["uuid"]
    
    me_action = None
    opp_action = None
    comm_cards = None
    
    if "preflop" in node.round_state["action_histories"]:
      for i in range(len(node.round_state["action_histories"]["preflop"])):
        data = node.round_state["action_histories"]["preflop"][i]
        if data["uuid"] == me_uuid:
          me_action = data["action"]
        elif data["uuid"] == opp_uuid:
          opp_action = data["action"]
          comm_cards = []
          
    if "flop" in node.round_state["action_histories"]:
      for i in range(len(node.round_state["action_histories"]["flop"])):
        data = node.round_state["action_histories"]["flop"][i]
        if data["uuid"] == me_uuid:
          me_action = data["action"]
        elif data["uuid"] == opp_uuid:
          opp_action = data["action"]
          comm_cards = node.round_state["community_card"][0:3]

    if "turn" in node.round_state["action_histories"]:
      for i in range(len(node.round_state["action_histories"]["turn"])):
        data = node.round_state["action_histories"]["turn"][i]
        if data["uuid"] == me_uuid:
          me_action = data["action"]
        elif data["uuid"] == opp_uuid:
          opp_action = data["action"]
          comm_cards = node.round_state["community_card"][0:4]

    if "river" in node.round_state["action_histories"]:
      for i in range(len(node.round_state["action_histories"]["river"])):
        data = node.round_state["action_histories"]["river"][i]
        if data["uuid"] == me_uuid:
          me_action = data["action"]
        elif data["uuid"] == opp_uuid:
          opp_action = data["action"]
          comm_cards = node.round_state["community_card"]
    
    if me_action in ["BIGBLIND", "SMALLBLIND"]:
      me_action = None
    if opp_action in ["BIGBLIND", "SMALLBLIND"]:
      opp_action = None
    if me_action == "CALL":
      me_action = "call"
    elif me_action == "RAISE":
      me_action = "raise"
    elif not (me_action is None):
      print("ERROR: me_action not well defined")
    if opp_action == "CALL":
      opp_action = "call"
    elif opp_action == "RAISE":
      opp_action = "raise"
    elif not (opp_action is None):
      print("ERROR: opp_action not well defined")

    dict["me_action"] = me_action
    dict["opp_action"] = opp_action
    dict["comm_cards"] = comm_cards

    return dict

  def mcts_search(self, node, timeout=0.18):
    timeout = time.time() + timeout
    best = node.actions[0]
    n = 0
    timer = time.time()
    opp_cards = None
    opp_bucket = None
    info = self.get_info(node)
    bucket_data = None
    count = None
    if not (info["opp_action"] is None):
      data = self.tally["actions"][(info["me_action"], info["opp_action"], info["pot"])]
      if data["no_obs"] > 8:
        count = data["count"].copy()
        for key in count.keys():
          count[key] = (count[key] + 0.0) / data["no_obs"]
        bucket_data =  self.buckets(info["comm_cards"], 10, cutdown=0.5)
    #print("Generating buckets took: " + str(time.time() - timer))
    while time.time() < timeout:
      timer = time.time()
      if not (info["opp_action"] is None):
        data = self.tally["actions"][(info["me_action"], info["opp_action"], info["pot"])]
        if data["no_obs"] > 8:
          rand_val = rand.random()
          random_bucket = None
          total = 0
          for k, v in count.items():
            total += v
            if rand_val <= total:
              random_bucket = k
              break
          if random_bucket <= 1:
            opp_bucket = 0
          elif random_bucket <= 4:
            opp_bucket = 1
          elif random_bucket <= 7:
            opp_bucket = 2
          else:
            opp_bucket = 3
          sample_frame = bucket_data[random_bucket]
          opp_cards = (rand.choice(sample_frame))["cards"]
      #print("Choosing opp cards took: " + str(time.time() - timer))

      #print("Opp cards are: ")
      #pprint.pprint(opp_cards)
      #print("Opp bucket is: ")
      #pprint.pprint(opp_bucket)      
      
      n += 1
      leaf = self.mcts_select(node, timeout, opp_bucket, opp_cards)
      if time.time() >= timeout:
        break
      child = self.mcts_expand(leaf)
      result = self.mcts_simulate(child, opp_cards)
      if time.time() >= timeout:
        break
      self.mcts_propogate(child, result)
      best = self.mcts_best(node)
    #print(n)
    for action, child in node.children.items():
      #print(action + ": " + str(child.expected))
      pass
    return best

  def mcts_select(self, node, timeout, opp_bucket=None, opp_cards = None):
    def uct(node):
      return node.expected + 200 * math.sqrt(math.log(node.parent.visited) / node.visited)
      
    if time.time() >= timeout:
      return None
    elif (not node.actions) or (len(node.actions) > len(node.children)):
      return node
    elif node.current_player == "chance":
      return self.mcts_select(rand.choice(list(node.children.values())), timeout, opp_bucket, opp_cards)
    elif (node.current_player == node.known_player) or (opp_bucket is None):
      maxi = None
      max_action = None
      for action, child in node.children.items():
        if maxi is None:
          max_action = action
          maxi = uct(child)
        elif uct(child) > maxi:
          max_action = action
          maxi = uct(child)
      return self.mcts_select(node.children[max_action], timeout, opp_bucket, opp_cards)
    else:
      info = self.get_info(node)
      opp_bucket = self.which_bucket_fast(opp_cards, node.round_state["community_card"])
      if opp_bucket <= 1:
        opp_bucket = 0
      elif opp_bucket <= 4:
        opp_bucket = 1
      elif opp_bucket <= 7:
        opp_bucket = 2
      else:
        opp_bucket = 3
      data = self.tally["cards"][(info["me_action"], opp_bucket, info["pot"])]
      if data["no_obs"] > 8:
        count = data["count"].copy()
        for key in count.keys():
          count[key] = (count[key] + 0.0) / data["no_obs"]
        rand_val = rand.random()
        random_action = None
        total = 0
        for k, v in count.items():
          total += v
          if rand_val <= total:
            random_action = k
            break
        if random_action in node.actions:
          #print("random action used in selection stage")
          return self.mcts_select(node.children[random_action], timeout, opp_bucket, opp_cards)
        elif random_action == "raise":
          return self.mcts_select(node.children["call"], timeout, opp_bucket, opp_cards)
        else:
          print("error: random action not defined.")
      else:
        maxi = None
        max_action = None
        for action, child in node.children.items():
          if maxi is None:
            max_action = action
            maxi = uct(child)
          elif uct(child) > maxi:
            max_action = action
            maxi = uct(child)
        return self.mcts_select(node.children[max_action], timeout, opp_bucket, opp_cards)

    
  def mcts_expand(self, node):
    if not node.actions:
      return node
    else:
      unexplored = node.actions[:]
      for action in unexplored:
        if action in node.children:
          unexplored.remove(action)
      action = rand.choice(unexplored)
      new_state = self.next_state(node.round_state, action)
      new_node = Group27Player.MCTSNode(new_state, node.hole_card, node.known_player, new_state["next_player"], node, self)
      node.children[action] = new_node
      return new_node
        
  def mcts_simulate(self, node, opp_cards=None):
    if node.round_state["street"] == "fold":
      fold_uuid = node.round_state["seats"][node.current_player]["uuid"]
      lost_amount = 0
      for k, v in node.round_state["action_histories"].items():
        for action in v:
          if action["uuid"] == fold_uuid and action["action"] in ["RAISE", "CALL"]:
            lost_amount += action["paid"]
      if node.current_player == node.round_state["small_blind_pos"]:
        lost_amount += 10
      else:
        lost_amount += 20
      if node.current_player == node.known_player:
        return -lost_amount
      elif node.current_player == (1 - node.known_player):
        return lost_amount
      else:
        print("error: node.current_player is not well-defined.")     
    else:  
      pot = node.round_state["pot"]["main"]["amount"]
      for k, v in node.round_state["action_histories"].items():
        if v and v[-1]["action"] == "RAISE":
          pot +=  v[-1]["add_amount"]
          break
      win = 0
      lose = 0
      community_cards = node.round_state["community_card"]
      hole_cards1 = node.hole_card
      remaining_cards = Group27Player.cards[:]
      for c in remaining_cards:
        if c in community_cards or c in hole_cards1:
          remaining_cards.remove(c)
      sample = rand.sample(remaining_cards, 2 + (5 - len(community_cards)))
      opponent_cards1 = sample[0:2] if (opp_cards is None) else opp_cards
      community_cards1 = community_cards + sample[2:]
      #if opp_cards:
        #print("opp cards used in simulation")
      if (HandEvaluator.eval_hand([Card.from_str(x) for x in hole_cards1],
                                 [Card.from_str(x) for x in community_cards1]) >
          HandEvaluator.eval_hand([Card.from_str(x) for x in opponent_cards1],
                                 [Card.from_str(x) for x in community_cards1])):
        win += 1
      else:
        lose += 1
      community_cards = node.round_state["community_card"]
      remaining_cards = Group27Player.cards[:]
      for c in remaining_cards:
        if c in community_cards:
          remaining_cards.remove(c)
      sample = rand.sample(remaining_cards, 4 + (5 - len(community_cards)))
      opponent_cards2 = sample[0:2] #if (opp_cards is None) else opp_cards
      hole_cards2 = sample[2:4]
      community_cards2 = community_cards + sample[4:]
      if (HandEvaluator.eval_hand([Card.from_str(x) for x in hole_cards2],
                                 [Card.from_str(x) for x in community_cards2]) >
          HandEvaluator.eval_hand([Card.from_str(x) for x in opponent_cards2],
                                 [Card.from_str(x) for x in community_cards2])):
        win += 1
      else:
        lose += 1
      if win > lose:
        return (pot + 0.0) / 2
      elif win == lose:
        return 0
      else:
        return -(pot + 0.0) / 2
    
  def mcts_propogate(self, node, result):
    multiplier = -1.0 if node.parent.current_player == (1 - node.known_player) else 1.0
    node.expected = (node.expected * node.visited + multiplier * result) / (node.visited + 1)
    node.visited += 1
    node = node.parent
    while node:
      if node.parent:
        if ((node.current_player == node.parent.current_player) or
            (node.current_player == "chance" and node.parent.current_player == node.known_player) or
            (node.parent.current_player == "chance" and node.current_player == node.known_player)):
          multiplier = 1.0
        else:
          multiplier = -1.0
        acc = 0
        visit = 0
        for k, v in node.children.items():
          acc += v.expected * multiplier * v.visited
          visit += v.visited
        node.expected = acc / visit
      node.visited += 1
      node = node.parent

  def mcts_best(self, node):
    maxi = None
    max_action = None
    for action, child in node.children.items():
      if maxi is None:
        max_action = action
        maxi = child.expected
      elif child.expected > maxi:
        max_action = action
        maxi = child.expected
    return max_action

  def next_state(self, state, action):
    if action == "raise":
      new_state = copy.deepcopy(state)
      uuid = new_state["seats"][new_state["next_player"]]["uuid"]
      raise_amount = 2 * new_state["small_blind_amount"] if (new_state["street"] in ["preflop", "flop"]) else 4 * new_state["small_blind_amount"]
      total_amount = (new_state["action_histories"][new_state["street"]][-1]["amount"]
                      if new_state["action_histories"][new_state["street"]]
                      else 0) + raise_amount
      prev_amount = 0
      for x in new_state["action_histories"][new_state["street"]]:
        if x["uuid"] == uuid:
          prev_amount = x["amount"]
      paid = total_amount - prev_amount
      new_state["action_histories"][new_state["street"]].append(
        {"action": "RAISE",
         "add_amount": raise_amount,
         "amount": total_amount,
         "paid": paid,
         "uuid": uuid})
      new_state["next_player"] = 1 - new_state["next_player"]
      new_state["pot"]["main"]["amount"] += paid
      new_state["seats"][new_state["next_player"]]["stack"] -= paid
      return new_state
    elif action == "call":
      new_state = copy.deepcopy(state)
      uuid = new_state["seats"][new_state["next_player"]]["uuid"]
      total_amount = (new_state["action_histories"][new_state["street"]][-1]["amount"]
                      if new_state["action_histories"][new_state["street"]]
                      else 0)
      prev_amount = 0
      for x in new_state["action_histories"][new_state["street"]]:
        if x["uuid"] == uuid:
          prev_amount = x["amount"]
      paid = total_amount - prev_amount
      new_state["action_histories"][new_state["street"]].append(
        {"action": "CALL",
         "amount": total_amount,
         "paid": paid,
         "uuid": uuid})
      new_state["pot"]["main"]["amount"] += paid
      new_state["seats"][new_state["next_player"]]["stack"] -= paid
      next_player = 1 - new_state["next_player"]
      next_player_uuid =  new_state["seats"][next_player]["uuid"]
      for x in new_state["action_histories"][new_state["street"]]:
        if x["uuid"] == next_player_uuid and x["action"] in ["RAISE", "CALL"]:
          next_player = "chance"
          break
      new_state["next_player"] = next_player
      if next_player == "chance":
        next_street = None
        if new_state["street"] == "preflop":
          next_street = "flop"
        elif new_state["street"] == "flop":
          next_street = "turn"
        elif new_state["street"] == "turn":
          next_street = "river"
        else:
          next_street = "showdown"
        new_state["action_histories"][next_street] = []
        new_state["street"] = next_street
      return new_state
    elif action == "fold":
      new_state = copy.deepcopy(state)
      new_state["street"] = "fold"
      return new_state
    else:
      new_state = copy.deepcopy(state)
      new_state["next_player"] = new_state["small_blind_pos"]
      new_state["community_card"].append(action)
      return new_state

  def next_actions(self, state, hole_cards):
    if state["street"] in ["fold", "showdown"]:
      return []
    if state["next_player"] == "chance":
      actions = Group27Player.cards[:]
      for card in actions:
        if card in state["community_card"] or card in hole_cards:
          actions.remove(card)
      return actions
    else:
      actions = ["call", "fold"]
      street_raise = 1 if state["street"] == "preflop" else 0
      player_raise = 0
      for street in list(state["action_histories"].keys()):
        for action in state["action_histories"][street]:
          if action["action"] == "RAISE":
            if street == state["street"]:
              street_raise += 1
            if action["uuid"] == state["seats"][state["next_player"]]["uuid"]:
              player_raise += 1
      if street_raise < 4 and player_raise < 4:
        actions.append("raise")
      return actions

  # returns a list of buckets of cards sorted by hand strength
  def buckets(self, comm_cards, number, cutdown=None):
    sample_space = Group27Player.cards[:]
    for card in sample_space:
      if card in comm_cards:
        sample_space.remove(card)
    samples = list(itertools.combinations(sample_space, 2))
    if cutdown:
      samples = rand.sample(samples, int(math.floor(len(samples) * cutdown)))
    samples2 = [{"cards": list(x), "value": HandEvaluator.eval_hand([Card.from_str(y) for y in list(x)], [Card.from_str(x) for x in comm_cards])} for x in samples]
    samples3 = sorted(samples2, key=lambda k: k['value'])
    length = len(samples3)
    group = length / number
    result = []
    end = 0
    while end < length:
      if len(result) == number:
        result[number - 1].extend(samples3[end:(end + group)])
      else:
        result.append(samples3[end:(end + group)])
      end += group

    if number == 10:
      for i in range(10):
        self.heuristic[i] = result[i][-1]["value"]
    return result

  # finds out which bucket a card is in based on hand strength
  def which_bucket(self, hole_cards, comm_cards, number):
    result = self.buckets(comm_cards, number)
    found = False
    bucket = -1
    for i in range(number):
      for x in result[i]:
        if set(x["cards"]) == set(hole_cards):
          found = True
          bucket = i
          break
      if found:
        break
    if bucket == -1:
      print("error! which_bucket returs -1!")
    return bucket

  # Faster version of above function that makes use of approximate bucketing based on hand strength
  def which_bucket_fast(self, hole_cards, comm_cards):
    result = HandEvaluator.eval_hand([Card.from_str(y) for y in hole_cards], [Card.from_str(x) for x in comm_cards])
    for key in self.heuristic:
      if result <= self.heuristic[key]:
        return key
    return 9

  # updates the tally for opponent modelling
  def update(self, key1, key2, value):
    data = self.tally[key1][key2]
    if data["round_avg"] is None:
      data["round_avg"] = self.round_count
      data["no_obs"] = 1
      data["count"][value] += 1
      #print("Incrementing: " + key1 + ", " + str(key2) + ": " + str(value))
    else:
      no_obs = data["no_obs"]
      round_avg = data["round_avg"]
      round_diff = self.round_count - round_avg
      round_factor = ((round_diff + 0.0) / self.round_count) * 0.1
      add_obs = max(1, math.floor(self.round_count / 30))
      backup = 0
      while (add_obs + 0.0) / no_obs > 0.4 and add_obs != 1:
        add_obs = max(1, math.floor(add_obs / 2))
        #print("add_obs is: " + str(add_obs))
        backup += 1
        if backup >= 100:
          add_obs = 1
          break
      data["no_obs"] += add_obs
      data["count"][value] += add_obs
      data["round_avg"] = (round_avg * no_obs + self.round_count * add_obs + 0.0) / (no_obs + add_obs)
      #print("Incrementing: " + key1 + ", " + str(key2) + ": " + str(value))
      #print("Prev round_avg: " + str(round_avg) + " new round avg: " + str(data["round_avg"]))
      #print("prev no obs: " + str(no_obs) + " add obs: " + str(add_obs))
      
  def __init__(self):
    BasePokerPlayer.__init__(self)
    self.who_am_i = -1
    self.round_count = 0
    self.heuristic = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    self.tally = {"actions": {(None, "call", 10): {"round_avg": None,
                                                   "no_obs": 0,
                                                   "count": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}},
                              (None, "call", 20): {"round_avg": None,
                                                   "no_obs": 0,
                                                   "count": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}},
                              (None, "call", 30): {"round_avg": None,
                                                   "no_obs": 0,
                                                   "count": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}},
                              (None, "call", 40): {"round_avg": None,
                                                   "no_obs": 0,
                                                   "count": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}},
                              (None, "raise", 10): {"round_avg": 1,
                                                    "no_obs": 8,
                                                    "count": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 2, 7: 2, 8: 2, 9: 2}},
                              (None, "raise", 20): {"round_avg": 1,
                                                    "no_obs": 8,
                                                    "count": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 2, 7: 2, 8: 2, 9: 2}},
                              (None, "raise", 30): {"round_avg": 1,
                                                    "no_obs": 8,
                                                    "count": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 2, 7: 2, 8: 2, 9: 2}},
                              (None, "raise", 40): {"round_avg": 1,
                                                    "no_obs": 8,
                                                    "count": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 2, 7: 2, 8: 2, 9: 2}},
                              ("call", "call", 10): {"round_avg": None,
                                                     "no_obs": 0,
                                                     "count": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}},
                              ("call", "call", 20): {"round_avg": None,
                                                     "no_obs": 0,
                                                     "count": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}},
                              ("call", "call", 30): {"round_avg": None,
                                                     "no_obs": 0,
                                                     "count": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}},
                              ("call", "call", 40): {"round_avg": None,
                                                     "no_obs": 0,
                                                     "count": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}},
                              ("call", "raise", 10): {"round_avg": 1,
                                                      "no_obs": 8,
                                                      "count": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 2, 7: 2, 8: 2, 9: 2}},
                              ("call", "raise", 20): {"round_avg": 1,
                                                      "no_obs": 8,
                                                      "count": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 2, 7: 2, 8: 2, 9: 2}},
                              ("call", "raise", 30): {"round_avg": 1,
                                                      "no_obs": 8,
                                                      "count": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 2, 7: 2, 8: 2, 9: 2}},
                              ("call", "raise", 40): {"round_avg": 1,
                                                      "no_obs": 8,
                                                      "count": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 2, 7: 2, 8: 2, 9: 2}},
                              ("raise", "call", 10): {"round_avg": None,
                                                      "no_obs": 0,
                                                      "count": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}},
                              ("raise", "call", 20): {"round_avg": None,
                                                      "no_obs": 0,
                                                      "count": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}},
                              ("raise", "call", 30): {"round_avg": None,
                                                      "no_obs": 0,
                                                      "count": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}},
                              ("raise", "call", 40): {"round_avg": None,
                                                      "no_obs": 0,
                                                      "count": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}},
                              ("raise", "raise", 10): {"round_avg": 1,
                                                       "no_obs": 8,
                                                       "count": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 2, 7: 2, 8: 2, 9: 2}},
                              ("raise", "raise", 20): {"round_avg": 1,
                                                       "no_obs": 8,
                                                       "count": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 2, 7: 2, 8: 2, 9: 2}},
                              ("raise", "raise", 30): {"round_avg": 1,
                                                       "no_obs": 8,
                                                       "count": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 2, 7: 2, 8: 2, 9: 2}},
                              ("raise", "raise", 40): {"round_avg": 1,
                                                       "no_obs": 8,
                                                       "count": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 2, 7: 2, 8: 2, 9: 2}}},
                  "cards": {(None, 0, 10): {"round_avg": None,
                                            "no_obs": 0,
                                            "count": {"fold": 0, "call": 0, "raise": 0}},
                            (None, 0, 20): {"round_avg": None,
                                            "no_obs": 0,
                                            "count": {"fold": 0, "call": 0, "raise": 0}},
                            (None, 0, 30): {"round_avg": None,
                                            "no_obs": 0,
                                            "count": {"fold": 0, "call": 0, "raise": 0}},
                            (None, 0, 40): {"round_avg": None,
                                            "no_obs": 0,
                                            "count": {"fold": 0, "call": 0, "raise": 0}},
                            (None, 1, 10): {"round_avg": None,
                                            "no_obs": 0,
                                            "count": {"fold": 0, "call": 0, "raise": 0}},
                            (None, 1, 20): {"round_avg": None,
                                            "no_obs": 0,
                                            "count": {"fold": 0, "call": 0, "raise": 0}},
                            (None, 1, 30): {"round_avg": None,
                                            "no_obs": 0,
                                            "count": {"fold": 0, "call": 0, "raise": 0}},
                            (None, 1, 40): {"round_avg": None,
                                            "no_obs": 0,
                                            "count": {"fold": 0, "call": 0, "raise": 0}},
                            (None, 2, 10): {"round_avg": None,
                                            "no_obs": 0,
                                            "count": {"fold": 0, "call": 0, "raise": 0}},
                            (None, 2, 20): {"round_avg": None,
                                            "no_obs": 0,
                                            "count": {"fold": 0, "call": 0, "raise": 0}},
                            (None, 2, 30): {"round_avg": None,
                                            "no_obs": 0,
                                            "count": {"fold": 0, "call": 0, "raise": 0}},
                            (None, 2, 40): {"round_avg": None,
                                            "no_obs": 0,
                                            "count": {"fold": 0, "call": 0, "raise": 0}},
                            (None, 3, 10): {"round_avg": 1,
                                            "no_obs": 8,
                                            "count": {"fold": 0, "call": 0, "raise": 8}},
                            (None, 3, 20): {"round_avg": 1,
                                            "no_obs": 8,
                                            "count": {"fold": 0, "call": 0, "raise": 8}},
                            (None, 3, 30): {"round_avg": 1,
                                            "no_obs": 8,
                                            "count": {"fold": 0, "call": 0, "raise": 8}},
                            (None, 3, 40): {"round_avg": 1,
                                            "no_obs": 8,
                                            "count": {"fold": 0, "call": 0, "raise": 8}},
                            ("call", 0, 10): {"round_avg": None,
                                              "no_obs": 0,
                                              "count": {"fold": 0, "call": 0, "raise": 0}},
                            ("call", 0, 20): {"round_avg": None,
                                              "no_obs": 0,
                                              "count": {"fold": 0, "call": 0, "raise": 0}},
                            ("call", 0, 30): {"round_avg": None,
                                              "no_obs": 0,
                                              "count": {"fold": 0, "call": 0, "raise": 0}},
                            ("call", 0, 40): {"round_avg": None,
                                              "no_obs": 0,
                                              "count": {"fold": 0, "call": 0, "raise": 0}},
                            ("call", 1, 10): {"round_avg": None,
                                              "no_obs": 0,
                                              "count": {"fold": 0, "call": 0, "raise": 0}},
                            ("call", 1, 20): {"round_avg": None,
                                              "no_obs": 0,
                                              "count": {"fold": 0, "call": 0, "raise": 0}},
                            ("call", 1, 30): {"round_avg": None,
                                              "no_obs": 0,
                                              "count": {"fold": 0, "call": 0, "raise": 0}},
                            ("call", 1, 40): {"round_avg": None,
                                              "no_obs": 0,
                                              "count": {"fold": 0, "call": 0, "raise": 0}},
                            ("call", 2, 10): {"round_avg": None,
                                              "no_obs": 0,
                                              "count": {"fold": 0, "call": 0, "raise": 0}},
                            ("call", 2, 20): {"round_avg": None,
                                              "no_obs": 0,
                                              "count": {"fold": 0, "call": 0, "raise": 0}},
                            ("call", 2, 30): {"round_avg": None,
                                              "no_obs": 0,
                                              "count": {"fold": 0, "call": 0, "raise": 0}},
                            ("call", 2, 40): {"round_avg": None,
                                              "no_obs": 0,
                                              "count": {"fold": 0, "call": 0, "raise": 0}},
                            ("call", 3, 10): {"round_avg": 1,
                                              "no_obs": 8,
                                              "count": {"fold": 0, "call": 0, "raise": 8}},
                            ("call", 3, 20): {"round_avg": 1,
                                              "no_obs": 8,
                                              "count": {"fold": 0, "call": 0, "raise": 8}},
                            ("call", 3, 30): {"round_avg": 1,
                                              "no_obs": 8,
                                              "count": {"fold": 0, "call": 0, "raise": 8}},
                            ("call", 3, 40): {"round_avg": 1,
                                              "no_obs": 8,
                                              "count": {"fold": 0, "call": 0, "raise": 8}},
                            ("raise", 0, 10): {"round_avg": None,
                                               "no_obs": 0,
                                               "count": {"fold": 0, "call": 0, "raise": 0}},
                            ("raise", 0, 20): {"round_avg": None,
                                               "no_obs": 0,
                                               "count": {"fold": 0, "call": 0, "raise": 0}},
                            ("raise", 0, 30): {"round_avg": None,
                                               "no_obs": 0,
                                               "count": {"fold": 0, "call": 0, "raise": 0}},
                            ("raise", 0, 40): {"round_avg": None,
                                               "no_obs": 0,
                                               "count": {"fold": 0, "call": 0, "raise": 0}},
                            ("raise", 1, 10): {"round_avg": None,
                                               "no_obs": 0,
                                               "count": {"fold": 0, "call": 0, "raise": 0}},
                            ("raise", 1, 20): {"round_avg": None,
                                               "no_obs": 0,
                                               "count": {"fold": 0, "call": 0, "raise": 0}},
                            ("raise", 1, 30): {"round_avg": None,
                                               "no_obs": 0,
                                               "count": {"fold": 0, "call": 0, "raise": 0}},
                            ("raise", 1, 40): {"round_avg": None,
                                               "no_obs": 0,
                                               "count": {"fold": 0, "call": 0, "raise": 0}},
                            ("raise", 2, 10): {"round_avg": None,
                                               "no_obs": 0,
                                               "count": {"fold": 0, "call": 0, "raise": 0}},
                            ("raise", 2, 20): {"round_avg": None,
                                               "no_obs": 0,
                                               "count": {"fold": 0, "call": 0, "raise": 0}},
                            ("raise", 2, 30): {"round_avg": None,
                                               "no_obs": 0,
                                               "count": {"fold": 0, "call": 0, "raise": 0}},
                            ("raise", 2, 40): {"round_avg": None,
                                               "no_obs": 0,
                                               "count": {"fold": 0, "call": 0, "raise": 0}},
                            ("raise", 3, 10): {"round_avg": None,
                                               "no_obs": 0,
                                               "count": {"fold": 0, "call": 0, "raise": 0}},
                            ("raise", 3, 20): {"round_avg": None,
                                               "no_obs": 0,
                                               "count": {"fold": 0, "call": 0, "raise": 0}},
                            ("raise", 3, 30): {"round_avg": None,
                                               "no_obs": 0,
                                               "count": {"fold": 0, "call": 0, "raise": 0}},
                            ("raise", 3, 40): {"round_avg": None,
                                               "no_obs": 0,
                                               "count": {"fold": 0, "call": 0, "raise": 0}}}}
   
  def declare_action(self, valid_actions, hole_card, round_state):
    time_start = time.time()
    #pprint.pprint(hole_card)
    self.who_am_i = round_state["next_player"]
    action = self.mcts_search(Group27Player.MCTSNode(round_state, hole_card, round_state["next_player"], round_state["next_player"], None, self))
    time_end = time.time()
    #print(time_end - time_start)
    actions = []
    for a in valid_actions:
      actions.append(a["action"])
    if not (action in actions):
      action = "call"
      print("Error, MCTS return invalid action")
    return action  # action returned here is sent to the poker engine

  def receive_game_start_message(self, game_info):
    # print("\n\n")
    # pprint.pprint(game_info)
    # print("---------------------------------------------------------------------")
    pass

  def receive_round_start_message(self, round_count, hole_card, seats):
    # print("My ID : "+self.uuid+", round count : "+str(round_count)+", hole card : "+str(hole_card))
    # pprint.pprint(seats)
    # print("-------------------------------")
    pass

  def receive_street_start_message(self, street, round_state):
    pass

  def receive_game_update_message(self, action, round_state):
    pass

  def receive_round_result_message(self, winners, hand_info, round_state):
    # print("My ID (round result) : "+self.uuid)
    # pprint.pprint(round_state)
    # print("\n\n")
    # self.round_count = self.round_count + 1
    start = time.time()
    self.round_count += 1
    if self.who_am_i == -1:
      me_name = winners[0]["name"]
      if round_state["seats"][0]["name"] == me_name:
       self.who_am_i = 0
      else:
        self.who_am_i = 1
    me_uuid = round_state["seats"][self.who_am_i]["uuid"]
    opp_uuid = round_state["seats"][1 - self.who_am_i]["uuid"]
    hole_cards = None
    if hand_info:
      for hand in hand_info:
        if hand["uuid"] == opp_uuid:
          hole_cards = hand["hand"]["card"]
    folder = None
    if (not hand_info) and (winners[0]["uuid"] == opp_uuid):
      folder = "ME"
    elif (not hand_info):
      folder = "OPP"
    bucket_1 = None
    bucket_2 = None
    if hand_info:
      bucket_1 = self.which_bucket(hole_cards, [], 10)
      if bucket_1 <= 1:
        bucket_2 = 0
      elif bucket_1 <= 4:
        bucket_2 = 1
      elif bucket_1 <= 7:
        bucket_2 = 2
      else:
        bucket_2 = 3

    pot = 0
    last_action = None

    for i in range(len(round_state["action_histories"]["preflop"])):
      data = round_state["action_histories"]["preflop"][i]

      if data["action"] in ["SMALLBLIND", "BIGBLIND"]:
        pot += data["amount"]
      elif "paid" in data:
        pot += data["paid"]

      pot_factor = (pot + 0.0) / round_state["small_blind_amount"]

      if pot_factor <= 10:
        pot_factor = 10
      elif pot_factor <= 20:
        pot_factor = 20
      elif pot_factor <= 30:
        pot_factor = 30
      else:
        pot_factor = 40

      if data["uuid"] == me_uuid:
        if data["action"] == "CALL":
          last_action = "call"
        elif data["action"] == "RAISE":
          last_action = "raise"

      if data["uuid"] == opp_uuid:
        if data["action"] == "FOLD":
          prev_action = round_state["action_histories"]["preflop"][i - 1]["action"]
          if prev_action == "BIGBLIND":
            self.update("cards", (None, 0, pot_factor), "fold")
            self.update("cards", (None, 1, pot_factor), "fold")
            pass
          elif prev_action == "CALL":
            self.update("cards", ("call", 0, pot_factor), "fold")
            self.update("cards", ("call", 1, pot_factor), "fold")
            pass
          elif prev_action == "RAISE":
            self.update("cards", ("raise", 0, pot_factor), "fold")
            self.update("cards", ("raise", 1, pot_factor), "fold")
            pass
          else:
            print("ERROR: fold action not preceded by valid action")
        if data["action"] == "CALL":
          prev_action = round_state["action_histories"]["preflop"][i - 1]["action"]
          if prev_action == "BIGBLIND":
            if folder == "OPP":
              self.update("cards", (None, 0, pot_factor), "call")
              self.update("cards", (None, 1, pot_factor), "call")
              self.update("actions", (None, "call", pot_factor), 0)
              self.update("actions", (None, "call", pot_factor), 1)
              self.update("actions", (None, "call", pot_factor), 2)
              self.update("actions", (None, "call", pot_factor), 3)
              self.update("actions", (None, "call", pot_factor), 4)
              pass
            elif folder == "ME":
              self.update("cards", (None, 2, pot_factor), "call")
              self.update("cards", (None, 3, pot_factor), "call")
              self.update("actions", (None, "call", pot_factor), 5)
              self.update("actions", (None, "call", pot_factor), 6)
              self.update("actions", (None, "call", pot_factor), 7)
              self.update("actions", (None, "call", pot_factor), 8)
              self.update("actions", (None, "call", pot_factor), 9)
              pass
            else:
              self.update("cards", (None, bucket_2, pot_factor), "call")
              self.update("actions", (None, "call", pot_factor), bucket_1)
          elif prev_action == "CALL":
            if folder == "OPP":
              self.update("cards", ("call", 0, pot_factor), "call")
              self.update("cards", ("call", 1, pot_factor), "call")
              self.update("actions", ("call", "call", pot_factor), 0)
              self.update("actions", ("call", "call", pot_factor), 1)
              self.update("actions", ("call", "call", pot_factor), 2)
              self.update("actions", ("call", "call", pot_factor), 3)
              self.update("actions", ("call", "call", pot_factor), 4)
              pass
            elif folder == "ME":
              self.update("cards", ("call", 2, pot_factor), "call")
              self.update("cards", ("call", 3, pot_factor), "call")
              self.update("actions", ("call", "call", pot_factor), 5)
              self.update("actions", ("call", "call", pot_factor), 6)
              self.update("actions", ("call", "call", pot_factor), 7)
              self.update("actions", ("call", "call", pot_factor), 8)
              self.update("actions", ("call", "call", pot_factor), 9)
              pass
            else:
              self.update("cards", ("call", bucket_2, pot_factor), "call")
              self.update("actions", ("call", "call", pot_factor), bucket_1)
          elif prev_action == "RAISE":
            if folder == "OPP":
              self.update("cards", ("raise", 0, pot_factor), "call")
              self.update("cards", ("raise", 1, pot_factor), "call")
              self.update("actions", ("raise", "call", pot_factor), 0)
              self.update("actions", ("raise", "call", pot_factor), 1)
              self.update("actions", ("raise", "call", pot_factor), 2)
              self.update("actions", ("raise", "call", pot_factor), 3)
              self.update("actions", ("raise", "call", pot_factor), 4)
              pass
            elif folder == "ME":
              self.update("cards", ("raise", 2, pot_factor), "call")
              self.update("cards", ("raise", 3, pot_factor), "call")
              self.update("actions", ("raise", "call", pot_factor), 5)
              self.update("actions", ("raise", "call", pot_factor), 6)
              self.update("actions", ("raise", "call", pot_factor), 7)
              self.update("actions", ("raise", "call", pot_factor), 8)
              self.update("actions", ("raise", "call", pot_factor), 9)
              pass
            else:
              self.update("cards", ("raise", bucket_2, pot_factor), "call")
              self.update("actions", ("raise", "call", pot_factor), bucket_1)
          else:
            print("ERROR: call action not preceded by valid action")
        if data["action"] == "RAISE":
          prev_action = round_state["action_histories"]["preflop"][i - 1]["action"]
          if prev_action == "BIGBLIND":
            if folder == "OPP":
              self.update("cards", (None, 0, pot_factor), "raise")
              self.update("cards", (None, 1, pot_factor), "raise")
              self.update("actions", (None, "raise", pot_factor), 0)
              self.update("actions", (None, "raise", pot_factor), 1)
              self.update("actions", (None, "raise", pot_factor), 2)
              self.update("actions", (None, "raise", pot_factor), 3)
              self.update("actions", (None, "raise", pot_factor), 4)
              pass
            elif folder == "ME":
              self.update("cards", (None, 2, pot_factor), "raise")
              self.update("cards", (None, 3, pot_factor), "raise")
              self.update("actions", (None, "raise", pot_factor), 5)
              self.update("actions", (None, "raise", pot_factor), 6)
              self.update("actions", (None, "raise", pot_factor), 7)
              self.update("actions", (None, "raise", pot_factor), 8)
              self.update("actions", (None, "raise", pot_factor), 9)
              pass
            else:
              self.update("cards", (None, bucket_2, pot_factor), "raise")
              self.update("actions", (None, "raise", pot_factor), bucket_1)
          elif prev_action == "CALL":
            if folder == "OPP":
              self.update("cards", ("call", 0, pot_factor), "raise")
              self.update("cards", ("call", 1, pot_factor), "raise")
              self.update("actions", ("call", "raise", pot_factor), 0)
              self.update("actions", ("call", "raise", pot_factor), 1)
              self.update("actions", ("call", "raise", pot_factor), 2)
              self.update("actions", ("call", "raise", pot_factor), 3)
              self.update("actions", ("call", "raise", pot_factor), 4)
              pass
            elif folder == "ME":
              self.update("cards", ("call", 2, pot_factor), "raise")
              self.update("cards", ("call", 3, pot_factor), "raise")
              self.update("actions", ("call", "raise", pot_factor), 5)
              self.update("actions", ("call", "raise", pot_factor), 6)
              self.update("actions", ("call", "raise", pot_factor), 7)
              self.update("actions", ("call", "raise", pot_factor), 8)
              self.update("actions", ("call", "raise", pot_factor), 9)
              pass
            else:
              self.update("cards", ("call", bucket_2, pot_factor), "raise")
              self.update("actions", ("call", "raise", pot_factor), bucket_1)
          elif prev_action == "RAISE":
            if folder == "OPP":
              self.update("cards", ("raise", 0, pot_factor), "raise")
              self.update("cards", ("raise", 1, pot_factor), "raise")
              self.update("actions", ("raise", "raise", pot_factor), 0)
              self.update("actions", ("raise", "raise", pot_factor), 1)
              self.update("actions", ("raise", "raise", pot_factor), 2)
              self.update("actions", ("raise", "raise", pot_factor), 3)
              self.update("actions", ("raise", "raise", pot_factor), 4)
              pass
            elif folder == "ME":
              self.update("cards", ("raise", 2, pot_factor), "raise")
              self.update("cards", ("raise", 3, pot_factor), "raise")
              self.update("actions", ("raise", "raise", pot_factor), 5)
              self.update("actions", ("raise", "raise", pot_factor), 6)
              self.update("actions", ("raise", "raise", pot_factor), 7)
              self.update("actions", ("raise", "raise", pot_factor), 8)
              self.update("actions", ("raise", "raise", pot_factor), 9)
              pass
            else:
              self.update("cards", ("raise", bucket_2, pot_factor), "raise")
              self.update("actions", ("raise", "raise", pot_factor), bucket_1)
          else:
            print("ERROR: raise action not preceded by valid action")

    if "flop" in round_state["action_histories"]:
      if hand_info:
        bucket_1 = self.which_bucket(hole_cards, round_state["community_card"][0:3], 10)
        if bucket_1 <= 1:
          bucket_2 = 0
        elif bucket_1 <= 4:
          bucket_2 = 1
        elif bucket_1 <= 7:
          bucket_2 = 2
        else:
          bucket_2 = 3

      for i in range(len(round_state["action_histories"]["flop"])):
        data = round_state["action_histories"]["flop"][i]

        if data["action"] in ["SMALLBLIND", "BIGBLIND"]:
          pot += data["amount"]
        elif "paid" in data:
          pot += data["paid"]

        pot_factor = (pot + 0.0) / round_state["small_blind_amount"]

        if pot_factor <= 10:
          pot_factor = 10
        elif pot_factor <= 20:
          pot_factor = 20
        elif pot_factor <= 30:
          pot_factor = 30
        else:
          pot_factor = 40

        if data["uuid"] == me_uuid:
          if data["action"] == "CALL":
            last_action = "call"
          elif data["action"] == "RAISE":
            last_action = "raise"

        if data["uuid"] == opp_uuid:
          if data["action"] == "FOLD":
            prev_action = None
            if i != 0:
              prev_action = round_state["action_histories"]["flop"][i - 1]["action"]
            if prev_action is None:
              self.update("cards", (last_action, 0, pot_factor), "fold")
              self.update("cards", (last_action, 1, pot_factor), "fold")
              pass
            elif prev_action == "CALL":
              self.update("cards", ("call", 0, pot_factor), "fold")
              self.update("cards", ("call", 1, pot_factor), "fold")
              pass
            elif prev_action == "RAISE":
              self.update("cards", ("raise", 0, pot_factor), "fold")
              self.update("cards", ("raise", 1, pot_factor), "fold")
              pass
            else:
              print("ERROR: fold action not preceded by valid action")
          if data["action"] == "CALL":
            prev_action = None
            if i != 0:
              prev_action = round_state["action_histories"]["flop"][i - 1]["action"]
            if prev_action is None:
              if folder == "OPP":
                self.update("cards", (last_action, 0, pot_factor), "call")
                self.update("cards", (last_action, 1, pot_factor), "call")
                self.update("actions", (last_action, "call", pot_factor), 0)
                self.update("actions", (last_action, "call", pot_factor), 1)
                self.update("actions", (last_action, "call", pot_factor), 2)
                self.update("actions", (last_action, "call", pot_factor), 3)
                self.update("actions", (last_action, "call", pot_factor), 4)
                pass
              elif folder == "ME":
                self.update("cards", (last_action, 2, pot_factor), "call")
                self.update("cards", (last_action, 3, pot_factor), "call")
                self.update("actions", (last_action, "call", pot_factor), 5)
                self.update("actions", (last_action, "call", pot_factor), 6)
                self.update("actions", (last_action, "call", pot_factor), 7)
                self.update("actions", (last_action, "call", pot_factor), 8)
                self.update("actions", (last_action, "call", pot_factor), 9)
                pass
              else:
                self.update("cards", (last_action, bucket_2, pot_factor), "call")
                self.update("actions", (last_action, "call", pot_factor), bucket_1)
            elif prev_action == "CALL":
              if folder == "OPP":
                self.update("cards", ("call", 0, pot_factor), "call")
                self.update("cards", ("call", 1, pot_factor), "call")
                self.update("actions", ("call", "call", pot_factor), 0)
                self.update("actions", ("call", "call", pot_factor), 1)
                self.update("actions", ("call", "call", pot_factor), 2)
                self.update("actions", ("call", "call", pot_factor), 3)
                self.update("actions", ("call", "call", pot_factor), 4)
                pass
              elif folder == "ME":
                self.update("cards", ("call", 2, pot_factor), "call")
                self.update("cards", ("call", 3, pot_factor), "call")
                self.update("actions", ("call", "call", pot_factor), 5)
                self.update("actions", ("call", "call", pot_factor), 6)
                self.update("actions", ("call", "call", pot_factor), 7)
                self.update("actions", ("call", "call", pot_factor), 8)
                self.update("actions", ("call", "call", pot_factor), 9)
                pass
              else:
                self.update("cards", ("call", bucket_2, pot_factor), "call")
                self.update("actions", ("call", "call", pot_factor), bucket_1)
            elif prev_action == "RAISE":
              if folder == "OPP":
                self.update("cards", ("raise", 0, pot_factor), "call")
                self.update("cards", ("raise", 1, pot_factor), "call")
                self.update("actions", ("raise", "call", pot_factor), 0)
                self.update("actions", ("raise", "call", pot_factor), 1)
                self.update("actions", ("raise", "call", pot_factor), 2)
                self.update("actions", ("raise", "call", pot_factor), 3)
                self.update("actions", ("raise", "call", pot_factor), 4)
                pass
              elif folder == "ME":
                self.update("cards", ("raise", 2, pot_factor), "call")
                self.update("cards", ("raise", 3, pot_factor), "call")
                self.update("actions", ("raise", "call", pot_factor), 5)
                self.update("actions", ("raise", "call", pot_factor), 6)
                self.update("actions", ("raise", "call", pot_factor), 7)
                self.update("actions", ("raise", "call", pot_factor), 8)
                self.update("actions", ("raise", "call", pot_factor), 9)
                pass
              else:
                self.update("cards", ("raise", bucket_2, pot_factor), "call")
                self.update("actions", ("raise", "call", pot_factor), bucket_1)
            else:
              print("ERROR: call action not preceded by valid action")
          if data["action"] == "RAISE":
            prev_action = None
            if i != 0:
              prev_action = round_state["action_histories"]["flop"][i - 1]["action"]
            if prev_action is None:
              if folder == "OPP":
                self.update("cards", (last_action, 0, pot_factor), "raise")
                self.update("cards", (last_action, 1, pot_factor), "raise")
                self.update("actions", (last_action, "raise", pot_factor), 0)
                self.update("actions", (last_action, "raise", pot_factor), 1)
                self.update("actions", (last_action, "raise", pot_factor), 2)
                self.update("actions", (last_action, "raise", pot_factor), 3)
                self.update("actions", (last_action, "raise", pot_factor), 4)
                pass
              elif folder == "ME":
                self.update("cards", (last_action, 2, pot_factor), "raise")
                self.update("cards", (last_action, 3, pot_factor), "raise")
                self.update("actions", (last_action, "raise", pot_factor), 5)
                self.update("actions", (last_action, "raise", pot_factor), 6)
                self.update("actions", (last_action, "raise", pot_factor), 7)
                self.update("actions", (last_action, "raise", pot_factor), 8)
                self.update("actions", (last_action, "raise", pot_factor), 9)
                pass
              else:
                self.update("cards", (last_action, bucket_2, pot_factor), "raise")
                self.update("actions", (last_action, "raise", pot_factor), bucket_1)
            elif prev_action == "CALL":
              if folder == "OPP":
                self.update("cards", ("call", 0, pot_factor), "raise")
                self.update("cards", ("call", 1, pot_factor), "raise")
                self.update("actions", ("call", "raise", pot_factor), 0)
                self.update("actions", ("call", "raise", pot_factor), 1)
                self.update("actions", ("call", "raise", pot_factor), 2)
                self.update("actions", ("call", "raise", pot_factor), 3)
                self.update("actions", ("call", "raise", pot_factor), 4)
                pass
              elif folder == "ME":
                self.update("cards", ("call", 2, pot_factor), "raise")
                self.update("cards", ("call", 3, pot_factor), "raise")
                self.update("actions", ("call", "raise", pot_factor), 5)
                self.update("actions", ("call", "raise", pot_factor), 6)
                self.update("actions", ("call", "raise", pot_factor), 7)
                self.update("actions", ("call", "raise", pot_factor), 8)
                self.update("actions", ("call", "raise", pot_factor), 9)
                pass
              else:
                self.update("cards", ("call", bucket_2, pot_factor), "raise")
                self.update("actions", ("call", "raise", pot_factor), bucket_1)
            elif prev_action == "RAISE":
              if folder == "OPP":
                self.update("cards", ("raise", 0, pot_factor), "raise")
                self.update("cards", ("raise", 1, pot_factor), "raise")
                self.update("actions", ("raise", "raise", pot_factor), 0)
                self.update("actions", ("raise", "raise", pot_factor), 1)
                self.update("actions", ("raise", "raise", pot_factor), 2)
                self.update("actions", ("raise", "raise", pot_factor), 3)
                self.update("actions", ("raise", "raise", pot_factor), 4)
                pass
              elif folder == "ME":
                self.update("cards", ("raise", 2, pot_factor), "raise")
                self.update("cards", ("raise", 3, pot_factor), "raise")
                self.update("actions", ("raise", "raise", pot_factor), 5)
                self.update("actions", ("raise", "raise", pot_factor), 6)
                self.update("actions", ("raise", "raise", pot_factor), 7)
                self.update("actions", ("raise", "raise", pot_factor), 8)
                self.update("actions", ("raise", "raise", pot_factor), 9)
                pass
              else:
                self.update("cards", ("raise", bucket_2, pot_factor), "raise")
                self.update("actions", ("raise", "raise", pot_factor), bucket_1)
            else:
              print("ERROR: raise action not preceded by valid action")

    if "turn" in round_state["action_histories"]:
      if hand_info:
        bucket_1 = self.which_bucket(hole_cards, round_state["community_card"][0:4], 10)
        if bucket_1 <= 1:
          bucket_2 = 0
        elif bucket_1 <= 4:
          bucket_2 = 1
        elif bucket_1 <= 7:
          bucket_2 = 2
        else:
          bucket_2 = 3
      for i in range(len(round_state["action_histories"]["turn"])):
        data = round_state["action_histories"]["turn"][i]

        if data["action"] in ["SMALLBLIND", "BIGBLIND"]:
          pot += data["amount"]
        elif "paid" in data:
          pot += data["paid"]

        pot_factor = (pot + 0.0) / round_state["small_blind_amount"]

        if pot_factor <= 10:
          pot_factor = 10
        elif pot_factor <= 20:
          pot_factor = 20
        elif pot_factor <= 30:
          pot_factor = 30
        else:
          pot_factor = 40

        if data["uuid"] == me_uuid:
          if data["action"] == "CALL":
            last_action = "call"
          elif data["action"] == "RAISE":
            last_action = "raise"

        if data["uuid"] == opp_uuid:
          if data["action"] == "FOLD":
            prev_action = None
            if i != 0:
              prev_action = round_state["action_histories"]["turn"][i - 1]["action"]
            if prev_action is None:
              self.update("cards", (last_action, 0, pot_factor), "fold")
              self.update("cards", (last_action, 1, pot_factor), "fold")
              pass
            elif prev_action == "CALL":
              self.update("cards", ("call", 0, pot_factor), "fold")
              self.update("cards", ("call", 1, pot_factor), "fold")
              pass
            elif prev_action == "RAISE":
              self.update("cards", ("raise", 0, pot_factor), "fold")
              self.update("cards", ("raise", 1, pot_factor), "fold")
              pass
            else:
              print("ERROR: fold action not preceded by valid action")
          if data["action"] == "CALL":
            prev_action = None
            if i != 0:
              prev_action = round_state["action_histories"]["turn"][i - 1]["action"]
            if prev_action is None:
              if folder == "OPP":
                self.update("cards", (last_action, 0, pot_factor), "call")
                self.update("cards", (last_action, 1, pot_factor), "call")
                self.update("actions", (last_action, "call", pot_factor), 0)
                self.update("actions", (last_action, "call", pot_factor), 1)
                self.update("actions", (last_action, "call", pot_factor), 2)
                self.update("actions", (last_action, "call", pot_factor), 3)
                self.update("actions", (last_action, "call", pot_factor), 4)
                pass
              elif folder == "ME":
                self.update("cards", (last_action, 2, pot_factor), "call")
                self.update("cards", (last_action, 3, pot_factor), "call")
                self.update("actions", (last_action, "call", pot_factor), 5)
                self.update("actions", (last_action, "call", pot_factor), 6)
                self.update("actions", (last_action, "call", pot_factor), 7)
                self.update("actions", (last_action, "call", pot_factor), 8)
                self.update("actions", (last_action, "call", pot_factor), 9)
                pass
              else:
                self.update("cards", (last_action, bucket_2, pot_factor), "call")
                self.update("actions", (last_action, "call", pot_factor), bucket_1)
            elif prev_action == "CALL":
              if folder == "OPP":
                self.update("cards", ("call", 0, pot_factor), "call")
                self.update("cards", ("call", 1, pot_factor), "call")
                self.update("actions", ("call", "call", pot_factor), 0)
                self.update("actions", ("call", "call", pot_factor), 1)
                self.update("actions", ("call", "call", pot_factor), 2)
                self.update("actions", ("call", "call", pot_factor), 3)
                self.update("actions", ("call", "call", pot_factor), 4)
                pass
              elif folder == "ME":
                self.update("cards", ("call", 2, pot_factor), "call")
                self.update("cards", ("call", 3, pot_factor), "call")
                self.update("actions", ("call", "call", pot_factor), 5)
                self.update("actions", ("call", "call", pot_factor), 6)
                self.update("actions", ("call", "call", pot_factor), 7)
                self.update("actions", ("call", "call", pot_factor), 8)
                self.update("actions", ("call", "call", pot_factor), 9)
                pass
              else:
                self.update("cards", ("call", bucket_2, pot_factor), "call")
                self.update("actions", ("call", "call", pot_factor), bucket_1)
            elif prev_action == "RAISE":
              if folder == "OPP":
                self.update("cards", ("raise", 0, pot_factor), "call")
                self.update("cards", ("raise", 1, pot_factor), "call")
                self.update("actions", ("raise", "call", pot_factor), 0)
                self.update("actions", ("raise", "call", pot_factor), 1)
                self.update("actions", ("raise", "call", pot_factor), 2)
                self.update("actions", ("raise", "call", pot_factor), 3)
                self.update("actions", ("raise", "call", pot_factor), 4)
                pass
              elif folder == "ME":
                self.update("cards", ("raise", 2, pot_factor), "call")
                self.update("cards", ("raise", 3, pot_factor), "call")
                self.update("actions", ("raise", "call", pot_factor), 5)
                self.update("actions", ("raise", "call", pot_factor), 6)
                self.update("actions", ("raise", "call", pot_factor), 7)
                self.update("actions", ("raise", "call", pot_factor), 8)
                self.update("actions", ("raise", "call", pot_factor), 9)
                pass
              else:
                self.update("cards", ("raise", bucket_2, pot_factor), "call")
                self.update("actions", ("raise", "call", pot_factor), bucket_1)
            else:
              print("ERROR: call action not preceded by valid action")
          if data["action"] == "RAISE":
            prev_action = None
            if i != 0:
              prev_action = round_state["action_histories"]["turn"][i - 1]["action"]
            if prev_action is None:
              if folder == "OPP":
                self.update("cards", (last_action, 0, pot_factor), "raise")
                self.update("cards", (last_action, 1, pot_factor), "raise")
                self.update("actions", (last_action, "raise", pot_factor), 0)
                self.update("actions", (last_action, "raise", pot_factor), 1)
                self.update("actions", (last_action, "raise", pot_factor), 2)
                self.update("actions", (last_action, "raise", pot_factor), 3)
                self.update("actions", (last_action, "raise", pot_factor), 4)
                pass
              elif folder == "ME":
                self.update("cards", (last_action, 2, pot_factor), "raise")
                self.update("cards", (last_action, 3, pot_factor), "raise")
                self.update("actions", (last_action, "raise", pot_factor), 5)
                self.update("actions", (last_action, "raise", pot_factor), 6)
                self.update("actions", (last_action, "raise", pot_factor), 7)
                self.update("actions", (last_action, "raise", pot_factor), 8)
                self.update("actions", (last_action, "raise", pot_factor), 9)
                pass
              else:
                self.update("cards", (last_action, bucket_2, pot_factor), "raise")
                self.update("actions", (last_action, "raise", pot_factor), bucket_1)
            elif prev_action == "CALL":
              if folder == "OPP":
                self.update("cards", ("call", 0, pot_factor), "raise")
                self.update("cards", ("call", 1, pot_factor), "raise")
                self.update("actions", ("call", "raise", pot_factor), 0)
                self.update("actions", ("call", "raise", pot_factor), 1)
                self.update("actions", ("call", "raise", pot_factor), 2)
                self.update("actions", ("call", "raise", pot_factor), 3)
                self.update("actions", ("call", "raise", pot_factor), 4)
                pass
              elif folder == "ME":
                self.update("cards", ("call", 2, pot_factor), "raise")
                self.update("cards", ("call", 3, pot_factor), "raise")
                self.update("actions", ("call", "raise", pot_factor), 5)
                self.update("actions", ("call", "raise", pot_factor), 6)
                self.update("actions", ("call", "raise", pot_factor), 7)
                self.update("actions", ("call", "raise", pot_factor), 8)
                self.update("actions", ("call", "raise", pot_factor), 9)
                pass
              else:
                self.update("cards", ("call", bucket_2, pot_factor), "raise")
                self.update("actions", ("call", "raise", pot_factor), bucket_1)
            elif prev_action == "RAISE":
              if folder == "OPP":
                self.update("cards", ("raise", 0, pot_factor), "raise")
                self.update("cards", ("raise", 1, pot_factor), "raise")
                self.update("actions", ("raise", "raise", pot_factor), 0)
                self.update("actions", ("raise", "raise", pot_factor), 1)
                self.update("actions", ("raise", "raise", pot_factor), 2)
                self.update("actions", ("raise", "raise", pot_factor), 3)
                self.update("actions", ("raise", "raise", pot_factor), 4)
                pass
              elif folder == "ME":
                self.update("cards", ("raise", 2, pot_factor), "raise")
                self.update("cards", ("raise", 3, pot_factor), "raise")
                self.update("actions", ("raise", "raise", pot_factor), 5)
                self.update("actions", ("raise", "raise", pot_factor), 6)
                self.update("actions", ("raise", "raise", pot_factor), 7)
                self.update("actions", ("raise", "raise", pot_factor), 8)
                self.update("actions", ("raise", "raise", pot_factor), 9)
                pass
              else:
                self.update("cards", ("raise", bucket_2, pot_factor), "raise")
                self.update("actions", ("raise", "raise", pot_factor), bucket_1)
            else:
              print("ERROR: raise action not preceded by valid action")

    if "river" in round_state["action_histories"]:
      if hand_info:
        bucket_1 = self.which_bucket(hole_cards, round_state["community_card"], 10)
        if bucket_1 <= 1:
          bucket_2 = 0
        elif bucket_1 <= 4:
          bucket_2 = 1
        elif bucket_1 <= 7:
          bucket_2 = 2
        else:
          bucket_2 = 3
          
      for i in range(len(round_state["action_histories"]["river"])):
        data = round_state["action_histories"]["river"][i]

        if data["action"] in ["SMALLBLIND", "BIGBLIND"]:
          pot += data["amount"]
        elif "paid" in data:
          pot += data["paid"]

        pot_factor = (pot + 0.0) / round_state["small_blind_amount"]

        if pot_factor <= 10:
          pot_factor = 10
        elif pot_factor <= 20:
          pot_factor = 20
        elif pot_factor <= 30:
          pot_factor = 30
        else:
          pot_factor = 40

        if data["uuid"] == me_uuid:
          if data["action"] == "CALL":
            last_action = "call"
          elif data["action"] == "RAISE":
            last_action = "raise"

        if data["uuid"] == opp_uuid:
          if data["action"] == "FOLD":
            prev_action = None
            if i != 0:
              prev_action = round_state["action_histories"]["river"][i - 1]["action"]
            if prev_action is None:
              self.update("cards", (last_action, 0, pot_factor), "fold")
              self.update("cards", (last_action, 1, pot_factor), "fold")
              pass
            elif prev_action == "CALL":
              self.update("cards", ("call", 0, pot_factor), "fold")
              self.update("cards", ("call", 1, pot_factor), "fold")
              pass
            elif prev_action == "RAISE":
              self.update("cards", ("raise", 0, pot_factor), "fold")
              self.update("cards", ("raise", 1, pot_factor), "fold")
              pass
            else:
              print("ERROR: fold action not preceded by valid action")
          if data["action"] == "CALL":
            prev_action = None
            if i != 0:
              prev_action = round_state["action_histories"]["river"][i - 1]["action"]
            if prev_action is None:
              if folder == "OPP":
                self.update("cards", (last_action, 0, pot_factor), "call")
                self.update("cards", (last_action, 1, pot_factor), "call")
                self.update("actions", (last_action, "call", pot_factor), 0)
                self.update("actions", (last_action, "call", pot_factor), 1)
                self.update("actions", (last_action, "call", pot_factor), 2)
                self.update("actions", (last_action, "call", pot_factor), 3)
                self.update("actions", (last_action, "call", pot_factor), 4)
                pass
              elif folder == "ME":
                self.update("cards", (last_action, 2, pot_factor), "call")
                self.update("cards", (last_action, 3, pot_factor), "call")
                self.update("actions", (last_action, "call", pot_factor), 5)
                self.update("actions", (last_action, "call", pot_factor), 6)
                self.update("actions", (last_action, "call", pot_factor), 7)
                self.update("actions", (last_action, "call", pot_factor), 8)
                self.update("actions", (last_action, "call", pot_factor), 9)
                pass
              else:
                self.update("cards", (last_action, bucket_2, pot_factor), "call")
                self.update("actions", (last_action, "call", pot_factor), bucket_1)
            elif prev_action == "CALL":
              if folder == "OPP":
                self.update("cards", ("call", 0, pot_factor), "call")
                self.update("cards", ("call", 1, pot_factor), "call")
                self.update("actions", ("call", "call", pot_factor), 0)
                self.update("actions", ("call", "call", pot_factor), 1)
                self.update("actions", ("call", "call", pot_factor), 2)
                self.update("actions", ("call", "call", pot_factor), 3)
                self.update("actions", ("call", "call", pot_factor), 4)
                pass
              elif folder == "ME":
                self.update("cards", ("call", 2, pot_factor), "call")
                self.update("cards", ("call", 3, pot_factor), "call")
                self.update("actions", ("call", "call", pot_factor), 5)
                self.update("actions", ("call", "call", pot_factor), 6)
                self.update("actions", ("call", "call", pot_factor), 7)
                self.update("actions", ("call", "call", pot_factor), 8)
                self.update("actions", ("call", "call", pot_factor), 9)
                pass
              else:
                self.update("cards", ("call", bucket_2, pot_factor), "call")
                self.update("actions", ("call", "call", pot_factor), bucket_1)
            elif prev_action == "RAISE":
              if folder == "OPP":
                self.update("cards", ("raise", 0, pot_factor), "call")
                self.update("cards", ("raise", 1, pot_factor), "call")
                self.update("actions", ("raise", "call", pot_factor), 0)
                self.update("actions", ("raise", "call", pot_factor), 1)
                self.update("actions", ("raise", "call", pot_factor), 2)
                self.update("actions", ("raise", "call", pot_factor), 3)
                self.update("actions", ("raise", "call", pot_factor), 4)
                pass
              elif folder == "ME":
                self.update("cards", ("raise", 2, pot_factor), "call")
                self.update("cards", ("raise", 3, pot_factor), "call")
                self.update("actions", ("raise", "call", pot_factor), 5)
                self.update("actions", ("raise", "call", pot_factor), 6)
                self.update("actions", ("raise", "call", pot_factor), 7)
                self.update("actions", ("raise", "call", pot_factor), 8)
                self.update("actions", ("raise", "call", pot_factor), 9)
                pass
              else:
                self.update("cards", ("raise", bucket_2, pot_factor), "call")
                self.update("actions", ("raise", "call", pot_factor), bucket_1)
            else:
              print("ERROR: call action not preceded by valid action")
          if data["action"] == "RAISE":
            prev_action = None
            if i != 0:
              prev_action = round_state["action_histories"]["river"][i - 1]["action"]
            if prev_action is None:
              if folder == "OPP":
                self.update("cards", (last_action, 0, pot_factor), "raise")
                self.update("cards", (last_action, 1, pot_factor), "raise")
                self.update("actions", (last_action, "raise", pot_factor), 0)
                self.update("actions", (last_action, "raise", pot_factor), 1)
                self.update("actions", (last_action, "raise", pot_factor), 2)
                self.update("actions", (last_action, "raise", pot_factor), 3)
                self.update("actions", (last_action, "raise", pot_factor), 4)
                pass
              elif folder == "ME":
                self.update("cards", (last_action, 2, pot_factor), "raise")
                self.update("cards", (last_action, 3, pot_factor), "raise")
                self.update("actions", (last_action, "raise", pot_factor), 5)
                self.update("actions", (last_action, "raise", pot_factor), 6)
                self.update("actions", (last_action, "raise", pot_factor), 7)
                self.update("actions", (last_action, "raise", pot_factor), 8)
                self.update("actions", (last_action, "raise", pot_factor), 9)
                pass
              else:
                self.update("cards", (last_action, bucket_2, pot_factor), "raise")
                self.update("actions", (last_action, "raise", pot_factor), bucket_1)
            elif prev_action == "CALL":
              if folder == "OPP":
                self.update("cards", ("call", 0, pot_factor), "raise")
                self.update("cards", ("call", 1, pot_factor), "raise")
                self.update("actions", ("call", "raise", pot_factor), 0)
                self.update("actions", ("call", "raise", pot_factor), 1)
                self.update("actions", ("call", "raise", pot_factor), 2)
                self.update("actions", ("call", "raise", pot_factor), 3)
                self.update("actions", ("call", "raise", pot_factor), 4)
                pass
              elif folder == "ME":
                self.update("cards", ("call", 2, pot_factor), "raise")
                self.update("cards", ("call", 3, pot_factor), "raise")
                self.update("actions", ("call", "raise", pot_factor), 5)
                self.update("actions", ("call", "raise", pot_factor), 6)
                self.update("actions", ("call", "raise", pot_factor), 7)
                self.update("actions", ("call", "raise", pot_factor), 8)
                self.update("actions", ("call", "raise", pot_factor), 9)
                pass
              else:
                self.update("cards", ("call", bucket_2, pot_factor), "raise")
                self.update("actions", ("call", "raise", pot_factor), bucket_1)
            elif prev_action == "RAISE":
              if folder == "OPP":
                self.update("cards", ("raise", 0, pot_factor), "raise")
                self.update("cards", ("raise", 1, pot_factor), "raise")
                self.update("actions", ("raise", "raise", pot_factor), 0)
                self.update("actions", ("raise", "raise", pot_factor), 1)
                self.update("actions", ("raise", "raise", pot_factor), 2)
                self.update("actions", ("raise", "raise", pot_factor), 3)
                self.update("actions", ("raise", "raise", pot_factor), 4)
                pass
              elif folder == "ME":
                self.update("cards", ("raise", 2, pot_factor), "raise")
                self.update("cards", ("raise", 3, pot_factor), "raise")
                self.update("actions", ("raise", "raise", pot_factor), 5)
                self.update("actions", ("raise", "raise", pot_factor), 6)
                self.update("actions", ("raise", "raise", pot_factor), 7)
                self.update("actions", ("raise", "raise", pot_factor), 8)
                self.update("actions", ("raise", "raise", pot_factor), 9)
                pass
              else:
                self.update("cards", ("raise", bucket_2, pot_factor), "raise")
                self.update("actions", ("raise", "raise", pot_factor), bucket_1)
            else:
              print("ERROR: raise action not preceded by valid action")        
    #pprint.pprint(winners)
    #pprint.pprint(hand_info)
    #pprint.pprint(round_state)
    #pprint.pprint(self.tally)
    print("round update: " + str(time.time() - start) + "seconds")
    
def setup_ai():
  return RandomPlayer()
