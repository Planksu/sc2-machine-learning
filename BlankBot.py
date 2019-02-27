import sc2
from sc2 import run_game, maps, Race, Difficulty, position, Result
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
        CYBERNETICSCORE, STALKER, STARGATE, VOIDRAY, ROBOTICSFACILITY, OBSERVER, ZEALOT
import random
import cv2
import os
import time
import numpy as np
import math
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard


os.environ["SC2PATH"] = 'E:/Blizzard Games/StarCraft II'
HEADLESS = False


class BlankBot(sc2.BotAI):
    def __init__(self, use_model=False, title=1, actorcritic=0):
        self.MAX_WORKERS = 50
        self.do_something_after = 0
        self.use_model = use_model
        self.title = title
        self.attack_unit_min = random.randrange(1,20)
        print(self.attack_unit_min)

        # key is unit tag, object is location
        self.scouts_and_spots = {}

        self.choices = {0: self.scout,
                        1: self.build_zealot,
                        2: self.build_gateway,
                        3: self.build_voidray,
                        4: self.build_stalker,
                        #5: self.build_worker,
                        #6: self.build_assimilator,
                        5: self.build_stargate,
                        #8: self.build_pylon,
                        6: self.attack,
                        #12: self.expand,
                        #13: self.do_nothing,
                        }

        self.train_data = []

        if self.use_model:
            print("USING MODEL!")
            #self.model = keras.models.load_model("BasicCNN-30-epochs-0.0001-LR-4.2")
            self.actorcritic = actorcritic

    def on_end(self, game_result):
        print('--- on_end called ---')

        if game_result == Result.Victory:
            np.save("train_data/{}.npy".format(str(int(time.time()))), np.array(self.train_data))

    async def build_scout(self):
        for rf in self.units(ROBOTICSFACILITY).ready.noqueue:
            print(len(self.units(OBSERVER)), self.time/3)
            if self.can_afford(OBSERVER) and self.supply_left > 0:
                await self.do(rf.train(OBSERVER))
                break
        if len(self.units(ROBOTICSFACILITY)) == 0:
            pylon = self.units(PYLON).ready.noqueue.random
            if self.units(CYBERNETICSCORE).ready.exists:
                if self.can_afford(ROBOTICSFACILITY) and not self.already_pending(ROBOTICSFACILITY):
                    await self.build(ROBOTICSFACILITY, near=pylon)


    async def on_step(self, iteration):
        #await self.scout()
        await self.expand()
        await self.intel()
        await self.distribute_workers()
        await self.build_worker()
        await self.build_pylon()
        await self.build_assimilator()
        await self.do_something()



    async def build_zealot(self):
        gateways = self.units(GATEWAY).ready.noqueue
        if gateways.exists:
            if self.can_afford(ZEALOT):
                await self.do(random.choice(gateways).train(ZEALOT))

    async def build_gateway(self):
        pylon = self.units(PYLON).ready.random
        if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
            await self.build(GATEWAY, near=pylon)

    async def build_voidray(self):
        stargates = self.units(STARGATE).ready
        if stargates.exists:
            if self.can_afford(VOIDRAY):
                await self.do(random.choice(stargates).train(VOIDRAY))

    async def build_stalker(self):
        pylon = self.units(PYLON).ready.random
        gateways = self.units(GATEWAY).ready
        cybernetics_cores = self.units(CYBERNETICSCORE).ready

        if gateways.exists and cybernetics_cores.exists:
            if self.can_afford(STALKER):
                await self.do(random.choice(gateways).train(STALKER))

        if not cybernetics_cores.exists:
            if self.units(GATEWAY).ready.exists:
                if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE, near=pylon)

    async def build_worker(self):
        nexuses = self.units(NEXUS).ready.noqueue
        if nexuses.exists:
            if self.can_afford(PROBE) and (len(self.units(PROBE)) < len(nexuses)*21) and len(self.units(PROBE)) < self.MAX_WORKERS:
                await self.do(random.choice(nexuses).train(PROBE))

    async def build_assimilator(self):
        for nexus in self.units(NEXUS).ready:
            vaspenes = self.state.vespene_geyser.closer_than(15.0, nexus)
            for vaspene in vaspenes:
                if not self.can_afford(ASSIMILATOR):
                    break
                worker = self.select_build_worker(vaspene.position)
                if worker is None:
                    break
                if not self.units(ASSIMILATOR).closer_than(1.0, vaspene).exists:
                    await self.do(worker.build(ASSIMILATOR, vaspene))

    async def build_stargate(self):
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random
            if self.units(CYBERNETICSCORE).ready.exists:
                if self.can_afford(STARGATE) and not self.already_pending(STARGATE):
                    await self.build(STARGATE, near=pylon)

    async def build_pylon(self):
        nexuses = self.units(NEXUS).ready
        if nexuses.exists:
            if self.can_afford(PYLON) and self.supply_left < 5:
                # build towards the center of the map rather than into the mineral line
                await self.build(PYLON, near=self.units(NEXUS).first.position.towards(self.game_info.map_center, 5))

    async def expand(self):
        try:
            if self.can_afford(NEXUS):
                await self.expand_now()
        except Exception as e:
            pass
            #print(str(e))

    async def do_nothing(self):
        wait = random.randrange(7, 100)/100
        self.do_something_after = self.time + wait

    def check_if_known_enemies(self):
        if len(self.known_enemy_units) > 0 or len(self.known_enemy_structures) > 0:
            return True
        else:
            return False

    def check_if_enough_army(self):
        army_count = len(self.units(ZEALOT))+len(self.units(STALKER))+len(self.units(VOIDRAY))
        if(army_count > self.attack_unit_min):
            return True
        else:
            return False

    def find_target(self):
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        else:
            return self.enemy_start_locations[0]

    async def attack(self):
        if self.check_if_known_enemies() and self.check_if_enough_army():
            print(self.check_if_enough_army())
            print(self.check_if_known_enemies())
            target = self.find_target()
            for u in self.units(VOIDRAY).idle:
                await self.do(u.attack(target))
            for u in self.units(STALKER).idle:
                await self.do(u.attack(target))
            for u in self.units(ZEALOT).idle:
                await self.do(u.attack(target))

    def random_location_variance(self, enemy_start_location):
        x = enemy_start_location[0]
        y = enemy_start_location[1]

        #  FIXED THIS
        x += random.randrange(-5, 5)
        y += random.randrange(-5, 5)

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.game_info.map_size[0]:
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            y = self.game_info.map_size[1]

        go_to = position.Point2(position.Pointlike((x,y)))

        return go_to

    async def scout(self):
        # key = distance to enemy start, object = location itself
        self.expand_dis_dir = {}
        for el in self.expansion_locations:
            distance_to_enemy_start = el.distance_to(self.enemy_start_locations[0])
            self.expand_dis_dir[distance_to_enemy_start] = el

        # since not python 3.7, have to do this to not mess up dictionary sorting
        self.ordered_exp_distances = sorted(k for k in self.expand_dis_dir)

        existing_ids = [unit.tag for unit in self.units]
        to_be_removed = []
        for noted_scout in self.scouts_and_spots:
            if noted_scout not in existing_ids:
                to_be_removed.append(noted_scout)

        for scout in to_be_removed:
            del self.scouts_and_spots[scout]

        if len(self.units(ROBOTICSFACILITY).ready) == 0:
            unit_type = PROBE
            unit_limit = 1
        else:
            unit_type = OBSERVER
            unit_limit = 15

        assign_scout = True

        if unit_type == PROBE:
            for unit in self.units(PROBE):
                if unit.tag in self.scouts_and_spots:
                    assign_scout = False

        if assign_scout:
            if len(self.units(unit_type).idle) > 0:
                for obs in self.units(unit_type).idle[:unit_limit]:
                    if obs.tag not in self.scouts_and_spots:
                        for dist in self.ordered_exp_distances:
                            try:
                                     # search dictionary for a value by key
                                location = self.expand_dis_dir[dist]
                                active_locations = [self.scouts_and_spots[k] for k in self.scouts_and_spots]
                                if location not in active_locations:
                                    if unit_type == PROBE:
                                        for unit in self.units(PROBE):
                                            if unit.tag in self.scouts_and_spots:
                                                continue
                                    await self.do(obs.move(location))
                                    self.scouts_and_spots[obs.tag] = location
                                    break
                            except Exception as e:
                                pass

        for obs in self.units(unit_type):
            if obs.tag in self.scouts_and_spots:
                if obs in [probe for probe in self.units(PROBE)]:
                    await self.do(obs.move(self.random_location_variance(self.scouts_and_spots[obs.tag])))

    async def intel(self):

        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)

        for unit in self.units().ready:
            pos = unit.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius*8), (255, 255, 255), math.ceil(int(unit.radius*0.5)))

        for unit in self.known_enemy_units:
            pos = unit.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius*8), (125, 125, 125), math.ceil(int(unit.radius*0.5)))

        try:
            line_max = 50
            mineral_ratio = self.minerals / 1500
            if mineral_ratio > 1.0:
                mineral_ratio = 1.0

            vespene_ratio = self.vespene / 1500
            if vespene_ratio > 1.0:
                vespene_ratio = 1.0

            population_ratio = self.supply_left / self.supply_cap
            if population_ratio > 1.0:
                population_ratio = 1.0

            plausible_supply = self.supply_cap / 200.0

            worker_weight = len(self.units(PROBE)) / (self.supply_cap-self.supply_left)
            if worker_weight > 1.0:
                worker_weight = 1.0

            cv2.line(game_data, (0, 19), (int(line_max*worker_weight), 19), (250, 250, 200), 3)  # worker/supply ratio
            cv2.line(game_data, (0, 15), (int(line_max*plausible_supply), 15), (220, 200, 200), 3)  # plausible supply (supply/200.0)
            cv2.line(game_data, (0, 11), (int(line_max*population_ratio), 11), (150, 150, 150), 3)  # population ratio (supply_left/supply)
            cv2.line(game_data, (0, 7), (int(line_max*vespene_ratio), 7), (210, 200, 0), 3)  # gas / 1500
            cv2.line(game_data, (0, 3), (int(line_max*mineral_ratio), 3), (0, 255, 25), 3)  # minerals minerals/1500
        except Exception as e:
            pass
            #print(str(e))

        # flip horizontally to make our final fix in visual representation:
        grayed = cv2.cvtColor(game_data, cv2.COLOR_BGR2GRAY)
        self.flipped = cv2.flip(grayed, 0)
        resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)
        if not HEADLESS:
            cv2.imshow(str(self.title), resized)
            cv2.waitKey(1)


    async def do_something(self):
        if self.time > self.do_something_after:
            if self.use_model:
                prediction = self.actorcritic.act()
                #prediction = self.model.predict([self.flipped.reshape([-1, 176, 200, 3])])
                choice = np.argmax(prediction[0])
            else:
                scout_weight = 1
                zealot_weight = 2
                gateway_weight = 1
                voidray_weight = 4
                stalker_weight = 4
                stargate_weight = 1
                attack_weight = 1

                choice_weights = scout_weight*[0]+zealot_weight*[1]+gateway_weight*[2]+voidray_weight*[3]+stalker_weight*[4]+stargate_weight*[5]+attack_weight*[6]
                choice = random.choice(choice_weights)
            try:
                await self.choices[choice]()
            except Exception as e:
                pass
                #print(str(e))
            y = np.zeros(7)
            y[choice] = 1
            self.train_data.append([y, self.flipped])




class ActorCritic:
    def __init__(self, sess):
        self.sess = sess
        self.memory = []

        self.learning_rate = 1e-5
        self.epsilon = 1.0
        self.epsilon_decay = .995
        self.gamma = .95
        self.tau = .125

        self.actor_state_input, self.actor_model = self.actor_model()
        _, self.target_actor_model = self.actor_model()

        self.actor_critic_grad = tf.placeholder(tf.float32, 
            [None, (176, 200, 1)]) 
        
        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output, 
            actor_model_weights, -self.actor_critic_grad)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize =  tf.train.AdamOptimizer(
            self.learning_rate).apply_gradients(grads)

        self.critic_state_input, self.critic_action_input, \
            self.critic_model = self.critic_model()
        _, _, self.target_critic_model = self.critic_model()

        self.critic_grads = tf.gradients(self.critic_model.output, 
            self.critic_action_input)
        
        # Initialize for later gradient calculations
        self.sess.run(tf.initialize_all_variables())


    def actor_model(self):
        model = Sequential()
        input_state = Input(shape=(176, 200, 1))
        model.add(Conv2D(32, (7, 7), padding='same',
        input_shape=(176,200,1),
                         activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3), padding='same',
                         activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (3, 3), padding='same',
                         activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

        learning_rate = 0.0001
        opt = keras.optimizers.adam(lr=learning_rate)#, decay=1e-5

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
        return input_state, model

    def critic_model(self):
        input_state = Input(shape=(176,200,1))
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(48)(state_h1)
        state_h3 = Dense(96)(state_h2)
        state_h4 = Dense(192)(state_h3)
        state_h5 = Dense(384)(state_h4)
            
        action_input = Input(shape=(176,200,1))
        action_h1    = Dense(384)(action_input)
            
        merged    = Add()([state_h5, action_h1])
        merged_h1 = Dense(24, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1)
        model  = Model(input=[state_input,action_input], 
                output=output)
            
        adam  = Adam(lr=0.0001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

    def remember(self, cur_state, action, reward, new_state, done):
        self.memory.append([cur_state, action, reward, new_state, done])

    def _train_critic(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, done = sample
            if not done:
                target_action = np.argmax(self.target_actor_model.predict(([self.flipped.reshape([-1, 176, 200, 3])]))[0])
                future_reward = np.argmax(self.target_critic_model.predict(([self.flipped.reshape([-1, 176, 200, 3])]))[0])
                reward += self.gamma * future_reward
            self.critic_model.fit([cur_state, action], reward, verbose=0)

    def _train_actor(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, _ = sample
            predicted_action = np.argmax(self.target_actor_model.predict(([self.flipped.reshape([-1, 176, 200, 3])]))[0])
            grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input:  cur_state,
                self.critic_action_input: predicted_action
            })[0]

            self.sess.run(self.optimize, feed_dict={
                self.actor_state_input: cur_state,
                self.actor_critic_grad: grads
            })

    def _update_actor_target(self):
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_critic_model.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]
        self.target_critic_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights  = self.critic_model.get_weights()
        critic_target_weights = self.critic_target_model.get_weights()
        
        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]
        self.critic_target_model.set_weights(critic_target_weights)

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()

    def act(self, cur_state):
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return self
        return np.argmax(self.target_actor_model.predict(([self.flipped.reshape([-1, 176, 200, 3])]))[0])

def main():
    sess = tf.Session()
    K.set_session(sess)
    actor_critic = ActorCritic(sess)

    num_trials = 10000
    trial_len = 500

    action = 0

    run_game(maps.get("AbyssalReefLE"), [
        Bot(Race.Protoss, BlankBot(use_model=False, actorcritic=actor_critic)),
        Computer(Race.Protoss, Difficulty.Easy),
        ], realtime=False)

    while True:
        actor_critic.update_target()
        

if __name__ == "__main__":
    main()
