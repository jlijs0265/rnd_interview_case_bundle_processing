# -*- coding: utf-8 -*-

import math
import time
import numpy as np
import json

import Common
from clustering import KMeans


MAX_NUM_NEIGHBORS = 4
NUM_WAYPOINTS_PER_CLUSTER, NUM_CLUSTERS_TO_CHECK_VICINITY = 2, 3
OUTLIER_PROPORTION = 0.3

PICKING_TIME = 30  # seconds
ARRIVING_DURATION = 180 * 1.1  # seconds
LEAVING_DURATION = 180 * 1.1  # seconds
SPEED = 18 * 0.9  # km/h


class Location:
    def __init__(self, lat, lng, est_arrival_at=0, est_departure_at=0):
        self.lat = float(lat)  # y
        self.lng = float(lng)  # x
        self.est_arrival_at = est_arrival_at
        self.est_departure_at = est_departure_at

    def __eq__(self, other):
        return self.lat == other.lat and self.lng == other.lng

    def distance(self, other):
        R = 6371  # km
        pi1 = math.radians(self.lat)
        pi2 = math.radians(other.lat)
        delta_pi = math.radians(other.lat - self.lat)
        delta_lambda = math.radians(other.lng - self.lng)

        a = math.sin(delta_pi * 0.5) * math.sin(delta_pi * 0.5) + \
            math.cos(pi1) * math.cos(pi2) * \
            math.sin(delta_lambda * 0.5) * math.sin(delta_lambda * 0.5)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        d = R * c
        return d


class WayPoint(Location):
    def __init__(self, delivery_dict):
        self.lat = delivery_dict['dropoffLat']
        self.lng = delivery_dict['dropoffLng']
        self.trip_type = delivery_dict['tripType']
        self.distribution_area_id = delivery_dict['dropoffDistributionAreaId']
        self.distribution_region_id = delivery_dict['dropoffDistributionRegionId']

        self.delivery_id = delivery_dict['deliveryId']
        self.delivery_number = delivery_dict['deliveryNumber']
        self.order_number = delivery_dict['orderNumber']

        self.item_volume = delivery_dict['itemVolume']

        self.distance_from_previous = None
        self.time_from_previous = None
        self.est_arrival_at = None
        self.est_departure_at = None

    def __str__(self):
        if self.est_arrival_at and self.time_from_previous:
            return '{} {:.0f}({:.0f})'.format(self.delivery_number, self.est_arrival_at, self.time_from_previous)
        else:
            return '{}'.format(self.delivery_number)

    def __repr__(self):
        return str(self)

    def update_distance_and_time(self, prev):
        self.distance_from_previous = prev.distance(self)
        self.time_from_previous = LEAVING_DURATION + self.distance_from_previous / SPEED * 3600 + ARRIVING_DURATION
        self.est_arrival_at = prev.est_arrival_at + self.time_from_previous
        self.est_departure_at = self.est_arrival_at + LEAVING_DURATION

    def to_dict(self):
        return {
            'deliveryId': self.delivery_id,
            'deliveryNumber': self.delivery_number,
            'orderNumber': self.order_number,
            'tripType': self.trip_type,
            'deliveryEta': self.est_arrival_at
        }


class Cluster(Location):
    def __init__(self, bundle, waypoints):
        self.bundle = bundle
        self.waypoints = waypoints
        self.lat = sum([waypoint.lat for waypoint in self.waypoints]) / len(self.waypoints)
        self.lng = sum([waypoint.lng for waypoint in self.waypoints]) / len(self.waypoints)

        self.dist_from_bundle = 0
        self.dist_from_others = 0
        self.cost = 0

    def __str__(self):
        return 'Cluster{}({}) {:.2f}'.format(self.bundle.bundle_number, self.bundle.bundle_id, self.cost)

    def __repr__(self):
        return str(self)

    def __lt__(self, other):
        return self.cost < other.cost

    def __gt__(self, other):
        return self.cost > other.cost

    def evaluate_cost(self):
        self.dist_from_bundle = self.distance(self.bundle)
        self.dist_from_others = sum(
            sorted([self.distance(other) for seq, other in enumerate(self.bundle.clusters) if other != self])
            [:NUM_CLUSTERS_TO_CHECK_VICINITY]
        )
        self.cost = self.dist_from_bundle + self.dist_from_others

    def evaluate_relocation(self, bundle):
        dist_from_bundle = self.distance(bundle)
        dist_from_others = sum(
            sorted([self.distance(other) for seq, other in enumerate(bundle.clusters) if other != self])
            [:NUM_CLUSTERS_TO_CHECK_VICINITY]
        )
        new_cost = dist_from_bundle + dist_from_others
        cost_diff = new_cost - self.cost
        return cost_diff


class Bundle(Location):
    def __init__(self, bundle_dict):
        self.bundle_id = bundle_dict['id']
        self.type = bundle_dict['type']
        self.bundle_number = bundle_dict['bundleNumber']
        self.delivery_partner_id = bundle_dict['deliveryPartnerId']
        self.departing_hub_id = bundle_dict['departingHubId']
        self.distribution_region_id = bundle_dict['distributionRegionId']
        self.waypoints = [WayPoint(delivery_dict) for delivery_dict in bundle_dict['deliveries']]
        self.lat = sum([waypoint.lat for waypoint in self.waypoints]) / len(self.waypoints)
        self.lng = sum([waypoint.lng for waypoint in self.waypoints]) / len(self.waypoints)
        self.distribution_area_ids = set([waypoint.distribution_area_id for waypoint in self.waypoints])
        self.etd = bundle_dict['etd']  # 배차 예상 시각
        self.etp = bundle_dict['etp']  # 픽업 예상 시각
        self.departing_hub = Location(
            bundle_dict['deliveries'][0]['pickupLat'],
            bundle_dict['deliveries'][0]['pickupLng']
        )

        # waypoints를 여러개의 클러스터로 분할하고, 그 중 인접도가 낮은 클러스터를 outliers로 관리
        self.clusters = []
        self.outliers = []
        self.cluster_waypoints()
        self.evaluate_clusters_and_sort()
        self.set_outliers()

        self.neighbors = []  # 인접 bundles
        self.addition = 0  # waypoints 변동 추적

    def __str__(self):
        to_show = 'waypoints'
        if to_show == 'len':
            return 'Bundle{}({}), Region{}, Partner{}, Delivery{}'.format(
                self.bundle_number, self.bundle_id, self.distribution_region_id, self.delivery_partner_id, len(self.waypoints)
            )
        elif to_show == 'waypoints':
            return 'Bundle{}({}), Region{}, Partner{}, {}'.format(
                self.bundle_number, self.bundle_id, self.distribution_region_id, self.delivery_partner_id, self.waypoints
            )

    def __repr__(self):
        return str(self)

    def __lt__(self, other):
        return self.addition < other.addition

    def __gt__(self, other):
        return self.addition > other.addition

    def cluster_waypoints(self):
        n_clusters = max(1, int(len(self.waypoints) / NUM_WAYPOINTS_PER_CLUSTER))
        labels, _, _, _ = KMeans().kmeans_with_locations(self.waypoints, n_clusters, max_iter=10000)
        clusters = {}
        for cluster_id, waypoint in zip(labels, self.waypoints):
            wp_in_cluster = clusters.setdefault(cluster_id, [])
            wp_in_cluster.append(waypoint)
        self.clusters = [Cluster(self, waypoints) for waypoints in clusters.values()]

    def evaluate_clusters_and_sort(self):
        [cluster.evaluate_cost() for cluster in self.clusters]
        self.clusters.sort(key=lambda cluster: cluster.cost, reverse=True)

    def set_outliers(self, proportion=OUTLIER_PROPORTION):
        num_outliers = math.ceil(len(self.clusters) * proportion)
        self.outliers = self.clusters[:num_outliers]

    def remove_cluster(self, cluster):
        self.clusters.remove(cluster)
        if cluster in self.outliers:
            self.outliers.remove(cluster)
        self.waypoints = [waypoint for waypoint in self.waypoints if waypoint not in cluster.waypoints]
        self.evaluate_clusters_and_sort()
        self.addition -= len(cluster.waypoints)

    def append_cluster(self, cluster):
        self.clusters.append(cluster)
        self.waypoints += cluster.waypoints
        self.evaluate_clusters_and_sort()
        self.addition += len(cluster.waypoints)

    def _flip_waypoints(self, idx_from, idx_to):
        waypoints_to_flip = self.waypoints[idx_from: idx_to + 1]
        waypoints_to_flip.reverse()
        self.waypoints = self.waypoints[:idx_from] + waypoints_to_flip + self.waypoints[idx_to + 1:]

    def sort_waypoints_with_two_opt(self):
        improved = True
        while improved:
            improved = False
            for flip_from in range(1, len(self.waypoints)):
                for flip_to in range(flip_from + 1, len(self.waypoints)):
                    # print('flip_from, flip_to: {}, {}'.format(flip_from, flip_to))
                    # print('switch (flip_from-1, flip_from) = ({}, {}) to (flip_from-1, flip_to) = ({}, {})'.format(flip_from-1, flip_from, flip_from-1, flip_to))
                    # print('switch (flip_to, flip_to+1) = ({}, {}) to (flip_from, flip_to+1) = ({}, {}) if edge exist'.format(flip_to, flip_to+1, flip_from, flip_to+1))
                    new_edge = self.waypoints[flip_from - 1].distance(self.waypoints[flip_to])
                    org_edge = self.waypoints[flip_from - 1].distance(self.waypoints[flip_from])
                    dist_diff = new_edge - org_edge
                    # print(1, new_edge, org_edge)
                    if flip_to + 1 < len(self.waypoints):  # if idx (flip_to + 1) exists on route.visits
                        new_edge = self.waypoints[flip_from].distance(self.waypoints[flip_to + 1])
                        org_edge = self.waypoints[flip_to].distance(self.waypoints[flip_to + 1])
                        # print(2, new_edge, org_edge)
                        dist_diff += new_edge - org_edge
                    if dist_diff < 0:
                        # print('{}, {} improved'.format(flip_from, flip_to))
                        improved = True
                        self._flip_waypoints(flip_from, flip_to)

    def update_waypoints_distance_and_time(self):
        hub_leaving_duration = LEAVING_DURATION + (len(self.waypoints) - 1) * PICKING_TIME
        self.departing_hub.est_departure_at = self.departing_hub.est_arrival_at + hub_leaving_duration
        self.etp = self.etd + self.departing_hub.est_departure_at
        prev = self.departing_hub
        for waypoint in self.waypoints:
            if prev is not None:
                waypoint.update_distance_and_time(prev)
            prev = waypoint

    def to_dict(self):
        return {
            'bundleId': self.bundle_id,
            'type': self.type,
            'bundleNumber': self.bundle_number,
            'deliveryPartnerId': self.delivery_partner_id,
            'departingHubId': self.departing_hub_id,
            'distributionRegionId': self.distribution_region_id,
            'distributionAreaIds': list(self.distribution_area_ids),
            'numberOfDeliveries': len(self.waypoints),
            'numberOfItems': len(self.waypoints),
            'volumeOfItems': sum([w.item_volume for w in self.waypoints]),
            'estTimeEnroute': self.waypoints[-1].est_arrival_at,
            'estDistanceEnroute': sum([w.distance_from_previous for w in self.waypoints]),
            'etd': self.etd,  # 배차 예상 시각
            'etp': self.etp,  # 픽업 예상 시각
            'etc': self.etd + self.waypoints[-1].est_arrival_at,  # 완료 예상 시각
            'deliveries': [w.to_dict() for w in self.waypoints]
        }


class BundleProcessor:
    def __init__(self, bundle_dict_list):
        self.bundles = [Bundle(bundle_dict) for bundle_dict in bundle_dict_list]
        self.sort_waypoints_then_update_distance_and_time_of_bundles()
        self.set_neighbors()

    def __str__(self):
        return '\n'.join([str(bundle) for bundle in self.bundles])

    def __repr__(self):
        return str(self)

    def set_neighbors(self):
        for bundle in self.bundles:
            others = [other for other in self.bundles if bundle != other]
            bundle.neighbors = sorted(others, key=lambda other: bundle.distance(other))[:MAX_NUM_NEIGHBORS]

    def get_all_outliers(self):
        outliers = []
        for bundle in self.bundles:
            outliers += bundle.outliers
        return sorted(outliers, reverse=True)

    def reset_clusters(self, outlier_proportion):
        for bundle in self.bundles:
            bundle.cluster_waypoints()
            bundle.evaluate_clusters_and_sort()
            bundle.set_outliers(outlier_proportion)

    def sort_waypoints_then_update_distance_and_time_of_bundles(self):
        for bundle in self.bundles:
            bundle.sort_waypoints_with_two_opt()
            bundle.update_waypoints_distance_and_time()

    def relocate_outliers(self):
        for outlier in self.get_all_outliers():
            cost_diffs = []
            for other_bundle in outlier.bundle.neighbors:
                cost_diffs.append(outlier.evaluate_relocation(other_bundle))
            if min(cost_diffs) < 0:
                other_bundle = outlier.bundle.neighbors[np.argmin(cost_diffs)]
                print('{} >> {} : {}'.format(outlier.bundle.bundle_number, other_bundle.bundle_number, len(outlier.waypoints)))
                outlier.bundle.remove_cluster(outlier)
                other_bundle.append_cluster(outlier)

    def plot_bundles(self, fn=None, plot_outliers=True):
        import matplotlib.pyplot as plt
        coords_by_bundle = []
        coords_by_cluster = []
        for bundle in self.bundles:
            coords = []
            for waypoint in bundle.waypoints:
                coords.append(np.array([waypoint.lat, waypoint.lng]))
            coords_by_bundle.append({
                'bundle': bundle,
                'coords': coords,
            })
        if plot_outliers:
            for bundle in self.bundles:
                for outlier in bundle.outliers:
                    coords = []
                    for waypoint in outlier.waypoints:
                        coords.append(np.array([waypoint.lat, waypoint.lng]))
                    coords_by_cluster.append({
                        'bundle': outlier.bundle,
                        'cluster': outlier,
                        'coords': coords,
                    })

        fig, ax = plt.subplots(figsize=(10, 10))
        for item in coords_by_bundle:
            bundle = item['bundle']
            coords = np.array(item['coords'])
            y, x = coords.transpose()
            ax.plot(x, y, marker='.', linestyle=':', label=bundle.bundle_number, alpha=1)

        for item in coords_by_cluster:
            bundle = item['bundle']
            cluster = item['cluster']
            coords = np.array(item['coords'])
            y, x = coords.transpose()
            ax.plot(x, y, marker='x', linestyle='-',
                    label='{} {}'.format(bundle.bundle_number, round(cluster.dist_from_bundle, 1)), alpha=1,
                    color='black')
        # plt.legend()
        if fn:
            plt.savefig(fn)
            plt.clf()
        else:
            plt.show()

    def process_bundles(self, max_iter=5, time_cutoff=10):
        started_at = time.time()
        self.plot_bundles(fn='./outputs/bundle_before.png', plot_outliers=False)
        for seq in range(max_iter):
            print('### iter: {}'.format(seq))
            self.relocate_outliers()
            if time.time() - started_at > time_cutoff:
                break
            self.reset_clusters(OUTLIER_PROPORTION)
        self.sort_waypoints_then_update_distance_and_time_of_bundles()
        self.plot_bundles(fn='./outputs/bundle_after.png', plot_outliers=False)
        return [bundle.to_dict() for bundle in self.bundles]


def load_sample_bundles(fn='./resources/sample_bundles.json'):
    with open(fn, 'r') as f:
        bundles = json.load(f)
    with open(fn, 'w') as f:
        json.dump(bundles, f)
    return bundles


if __name__ == '__main__':
    bundles = load_sample_bundles()
    bp = BundleProcessor(bundles)
    org_bundles = [bundle.to_dict() for bundle in bp.bundles]
    processed_bundles = bp.process_bundles(max_iter=5, time_cutoff=10)
    print(org_bundles)
    print(processed_bundles)
