import sc2
from sc2 import run_game, maps, Race, Difficulty, position, Result
from sc2.data import PlayerType
from sc2.units import Units
from sc2.unit import Unit
from sc2.ids.unit_typeid import UnitTypeId
from sc2.player import Bot, Computer, Human, Player
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
        CYBERNETICSCORE, STALKER, STARGATE, VOIDRAY, ROBOTICSFACILITY, OBSERVER, ZEALOT, IMMORTAL
from sc2.unit import AbilityId
import random
import cv2
import os
import time
import numpy as np
import math
#import keras
#from keras import backend as K


os.environ["SC2PATH"] = 'D:/Blizzard Games/StarCraft II'
HEADLESS = False

class BlankBot(sc2.BotAI):
    def __init__(self, use_model=False, title=1):
        self.use_model = use_model
        self.title = title
        self.attacking_units = []
        self.max_workers = 57 #19 * 3 = 3 bases workers
        self.current_second = 0
        self.attacker_update_delay = 0

        # key is unit tag, object is location
        self.scouts_and_spots = {}

        self.choices = {#0: self.scout,
                        0: self.build_zealot,
                        1: self.build_gateway,
                        #2: self.build_voidray,
                        2: self.build_stalker,
                        3: self.build_immortal,
                        #5: self.build_worker,
                        4: self.build_assimilator,
                        #5: self.build_stargate,
                        #8: self.build_pylon,
                        5: self.attack,
                        6: self.expand,
                        7: self.defend,
                        }

        self.train_data = []

        if self.use_model:
            print("USING MODEL!")
            self.model = keras.models.load_model("<model name>")

    async def on_step(self, iteration):
        # This if is here to restrict the APM of the bot to 1 per second
        if(self.time > self.current_second):
            # Intel method saves the map status to game_data
            await self.intel()
            await self.distribute_workers()
            await self.scout()

            # Only do these if we have enough minerals, to save some cpu time
            if self.minerals >= 100:
                await self.build_worker()
                await self.build_pylon()

            # Attack if our supply is full
            if self.supply_left <= 5 and self.supply_cap >= 195:
                await self.attack()

            # Finally, do something
            await self.do_something()

            # Increment the current second
            self.current_second = self.current_second + 1
        elif(self.time > self.attacker_update_delay):
            # The APM for microing military units is 120
            await self.update_attackers()
            self.attacker_update_delay = self.attacker_update_delay + 0.5


    def on_end(self, game_result):
        print('--- on_end called ---')

        with open("gameout-random-vs-easy.txt","a") as f:
            if self.use_model:
                f.write("Model {}\n".format(game_result))
            else:
                f.write("Random {}\n".format(game_result))

        if game_result == Result.Victory:
            np.save("train_data/{}.npy".format(str(int(time.time()))), np.array(self.train_data))

    async def on_unit_destroyed(self, unit_tag):
        for u in self.attacking_units:
            if unit_tag == u.tag:
                self.attacking_units.remove(u)
                break

    async def update_attackers(self):
        if len(self.attacking_units) > 0:
            for u in self.attacking_units:
                target = self.find_target(u)
                await self.do(u.attack(target))

    async def build_scout(self):
        for rf in self.units(ROBOTICSFACILITY).ready.noqueue:
            if self.can_afford(OBSERVER) and self.supply_left > 0:
                await self.do(rf.train(OBSERVER))
                break
        if len(self.units(ROBOTICSFACILITY)) == 0:
            pylon = self.units(PYLON).ready.noqueue.random
            if self.units(CYBERNETICSCORE).ready.exists:
                if self.can_afford(ROBOTICSFACILITY) and not self.already_pending(ROBOTICSFACILITY):
                    await self.build(ROBOTICSFACILITY, near=pylon.position.towards(self.game_info.map_center,  5))

    async def build_zealot(self):
        gateways = self.units(GATEWAY).ready.noqueue
        if gateways.exists:
            if self.can_afford(ZEALOT):
                await self.do(random.choice(gateways).train(ZEALOT))

    async def build_gateway(self):
        pylon = self.units(PYLON).ready.random
        if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
            await self.build(GATEWAY, near=pylon.position.towards(self.game_info.map_center, 5))

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
                    await self.build(CYBERNETICSCORE, near=pylon.position.towards(self.game_info.map_center, 5))

    async def build_worker(self):
        nexuses = self.units(NEXUS).ready.noqueue
        if nexuses.exists:
            if self.can_afford(PROBE) and len(self.units(PROBE)) < self.max_workers:
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
                    await self.build(STARGATE, near=pylon.position.towards(self.game_info.map_center,5))

    async def build_pylon(self):
        nexuses = self.units(NEXUS).ready
        if nexuses.exists:
            if self.can_afford(PYLON) and self.supply_left < 5:
                # build towards the center of the map rather than into the mineral line
                await self.build(PYLON, near=self.units(NEXUS).first.position.towards(self.game_info.map_center,5))

    async def build_immortal(self):
        robos = self.units(ROBOTICSFACILITY).ready.noqueue
        if len(robos) > 0:
            if self.can_afford(IMMORTAL):
                await self.do(random.choice(robos).train(IMMORTAL))
        else:
            pylon = self.units(PYLON).ready.noqueue.random
            if self.units(CYBERNETICSCORE).ready.exists:
                if self.can_afford(ROBOTICSFACILITY) and not self.already_pending(ROBOTICSFACILITY):
                    await self.build(ROBOTICSFACILITY, near=pylon.position.towards(self.game_info.map_center, 5))


    async def expand(self):
        try:
            if self.can_afford(NEXUS):
                await self.expand_now()
        except Exception as e:
            pass

    def check_if_known_enemies(self):
        if len(self.known_enemy_units) > 0 or len(self.known_enemy_structures) > 0:
            return True
        else:
            return False

    def find_target(self, this_unit):
        if this_unit.weapon_cooldown <= self._client.game_step / 2:
            if len(self.known_enemy_units) > 0:
                enemies_in_range = self.known_enemy_units.filter(lambda u: this_unit.target_in_range(u))
                if len(enemies_in_range) > 0:
                    filtered_enemies_in_range = enemies_in_range.of_type(UnitTypeId.IMMORTAL)
                    if len(filtered_enemies_in_range) > 0:
                        target = min(filtered_enemies_in_range, key=lambda u: u.health)
                    if not filtered_enemies_in_range:
                        filtered_enemies_in_range = enemies_in_range.of_type(UnitTypeId.VOIDRAY)
                        if len(filtered_enemies_in_range) > 0:
                            target = min(filtered_enemies_in_range, key=lambda u: u.health)
                        if not filtered_enemies_in_range:
                            filtered_enemies_in_range = enemies_in_range.of_type(UnitTypeId.STALKER)
                            if filtered_enemies_in_range:
                                target = min(filtered_enemies_in_range, key=lambda u: u.health)
                            if not filtered_enemies_in_range:
                                target = min(enemies_in_range, key=lambda u: u.health)
                    return target
                elif len(self.known_enemy_units) > 0:
                    return self.known_enemy_units.closest_to(this_unit)
                elif len(self.known_enemy_structures) > 0:
                    return self.known_enemy_structures.closest_to(this_unit)
                else:
                    return self.get_next_expansion()

    async def attack(self):
        print("this is the start of the attack method")
        voidrays = self.units.filter(lambda unit: unit.type_id==VOIDRAY)
        stalkers = self.units.filter(lambda unit: unit.type_id==STALKER)
        zealots = self.units.filter(lambda unit: unit.type_id==ZEALOTS)
        immortals = self.units.filter(lambda unit: unit.type_id==IMMORTAL)

        print("length" + len(zealots))
        print(len(stalkers))
        print(len(voidrays))
        print(len(immortals))


        for v in voidrays:
            if v not in self.attacking_units:
                self.attacking_units.append(v)
                await self.do(v.attack(self.find_target(v)))
        for s in stalkers:
            if s not in self.attacking_units:
                self.attacking_units.append(s)
                await self.do(s.attack(self.find_target(s)))
        for z in zealots:
            if z not in self.attacking_units:
                self.attacking_units.append(z)
                await self.do(z.attack(self.find_target(z)))
        for i in immortals:
            if i not in self.attacking_units:
                self.attacking_units.append(i)
                await self.do(i.attack(self.find_target(i)))


    async def defend(self):
        if self.check_if_known_enemies():
            voidrays = self.units.filter(lambda unit: unit.type_id==VOIDRAY and unit.noqueue)
            stalker = self.units.filter(lambda unit: unit.type_id==STALKER and unit.noqueue)
            zealot = self.units.filter(lambda unit: unit.type_id==ZEALOT and unit.noqueue)
            immortals = self.units.filter(lambda unit: unit.type_id==IMMORTAL and unit.noqueue)

            for u in self.known_enemy_units:
                if(u.distance_to(self.enemy_start_locations[0]) > u.distance_to(self.start_location)):
                    target = u.position
                    for v in voidrays:
                        await self.do(v.attack(target))
                    for s in stalkers:
                        await self.do(s.attack(target))
                    for z in zealots:
                        await self.do(z.attack(target))
                    for i in immortals:
                        await self.do(i.attack(target))


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

        # flip horizontally to make our final fix in visual representation:
        grayed = cv2.cvtColor(game_data, cv2.COLOR_BGR2GRAY)
        self.flipped = cv2.flip(grayed, 0)
        resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)
        if not HEADLESS:
            cv2.imshow(str(self.title), resized)
            cv2.waitKey(1)


    async def do_something(self):
        if self.use_model:
            prediction = self.model.predict([self.flipped.reshape([-1, 176, 200, 3])])
            choice = np.argmax(prediction[0])
        else:
            zealot = 10
            gateway = 15
            stalker = 20
            immortal = 10
            assimilator = 5
            attack = 1
            expand = 40
            defend = 15

            choice_weights = zealot*[0]+gateway*[1]+stalker*[2]+immortal*[3]+assimilator*[4]+attack*[5]+expand*[6]+defend*[7]
            choice = random.choice(choice_weights)

            #print(choice)
        try:
            await self.choices[choice]()
        except Exception as e:
            pass
                #print(str(e))
        y = np.zeros(8)
        y[choice] = 1
        self.train_data.append([y, self.flipped])


run_game(maps.get("AbyssalReefLE"), [
        Bot(Race.Protoss, BlankBot(use_model=False)),
        Computer(Race.Protoss, Difficulty.Medium),
        ], realtime=False)
