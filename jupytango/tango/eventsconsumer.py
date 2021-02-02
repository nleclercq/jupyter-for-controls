from __future__ import print_function
import os
import logging

try:
    # just in case tango stuffs are not installed
    from PyTango import EventType, EventData, DeviceProxy, AttrConfEventData, DataReadyEventData
except:
    pass

from jupytango.task import Task


class TangoEventSubscriptionAction(object):
    """a Tango event subscription form"""

    # action type
    DO_NOTHING = 0
    AUTO_CONFIGURE = 1

    # lowest polling period allowed in ms
    MIN_POLLING_THRESHOLD_IN_MS = 100

    def __init__(self, action_type=DO_NOTHING):
        self.type = action_type  # do nothing
        self.polling_period_ms = 1000  # polling thread period on server side
        self.event_period_ms = 1000  # periodic event period in ms
        self.absolute_change = 0.0  # change event absolute variation
        self.relative_change = 0.0  # change event relative variation


class TangoEventSubscriptionForm(object):
    """a Tango event subscription form"""

    def __init__(self, **kwargs):
        self.dev = kwargs.pop('dev', None)  # event source: [fully qualified] device name
        self.attr = kwargs.pop('attr', None)  # event source: attribute name
        self.evt_type = kwargs.pop('evt_type', None)  # tango event type (see PyTango.EventType)
        self.message_id = kwargs.pop('message_id', None)  # associated Task message id
        self.user_data = kwargs.pop('user_data', None)  # user data: any object (i.e. data) to be attached to the event
        self.action = kwargs.pop('action', None)  # action to perform at subscription
        if kwargs:
            raise ValueError('unknown keyword arguments: {}'.format(list(kwargs)))


class TangoEventInfo(object):
    """a Tango event subscription holder"""

    def __init__(self, **kwargs):
        self.attr = kwargs.pop('attr', None)  # event source: attribute name
        self.dev_proxy = kwargs.pop('dev_proxy', None)  # event source: device proxy
        self.evt_type = kwargs.pop('evt_type', None)  # tango event type (see PyTango.EventType)
        self.message_id = kwargs.pop('message_id', None)  # associated EventConsumer message id (if relevant)
        self.user_data = kwargs.pop('user_data', None)  # user data: any object (i.e. data) to be attached to the event
        self.__evt_id = None  # filled at subscription
        if kwargs:
            raise ValueError('unknown keyword arguments: {}'.format(list(kwargs)))

    def subscribe(self, callback):
        if self.__evt_id is not None:
            raise RuntimeError('event already subscribed')
        self.__evt_id = self.dev_proxy.subscribe_event(self.attr, self.evt_type, callback)

    def unsubscribe(self):
        if self.__evt_id is not None:
            try:
                self.dev_proxy.unsubscribe_event(self.__evt_id)
            except:
                pass
            finally:
                self.__evt_id = None

    def dump(self):
        print("event.info.source.......{}".format(self.attr))
        print("event.info.evt_type.....{}".format(self.evt_type))
        print("event.info.message_id...{}".format(self.message_id))
        print("event.info.user_data....{}".format(self.user_data))
        print("event.info.evt_id.......{}".format(self.__evt_id))


class TangoEvent(object):
    """a Tango event holder"""

    def __init__(self, **kwargs):
        self.info = kwargs.pop('info', TangoEventInfo())  # event (subscription) info
        self.data = kwargs.pop('data', None)  # instance of EventData, AttrConfEventData or DataReadyEventData
        if kwargs:
            raise ValueError('unknown keyword arguments: {}'.format(list(kwargs)))

    def dump(self):
        self.info.dump()
        print("tango_event_type..........{}".format(event_data_to_event_type(self.data)))
        print("tango.event-data.source...{}".format(self.data.attr_name))
        print("tango.event-data.event....{}".format(self.data.event))
        print("tango.event-data.err......{}".format(self.data.err))
        print("tango.event-data.errors...{}".format(self.data.errors))


def event_source_str(fqan):
    if fqan.find('tango://') != -1:
        return
    try:
        tango_hosts_str = os.environ['TANGO_HOST']
    except KeyError:
        raise KeyError('no TANGO_HOST env. var. defined!')
    tango_hosts = tango_hosts_str.split(",")
    if len(tango_hosts) == 0:
        raise KeyError('no TANGO_HOST env. var. defined!')
    tango_host = tango_hosts[0]
    return "tango://{}/{}".format(tango_host.lower(), fqan.lower())


def event_data_to_event_type(data):
    if isinstance(data, EventData):
        if data.event == "change":
            return EventType.CHANGE_EVENT
        elif data.event == "user_event":
            return EventType.USER_EVENT
        return EventType.PERIODIC_EVENT
    elif isinstance(data, AttrConfEventData):
        return EventType.ATTR_CONF_EVENT
    elif isinstance(data, DataReadyEventData):
        return EventType.DATA_READY_EVENT
    raise TypeError("unsupported Tango event data type: {!r}".format(data))


def pre_subscription_action(form, proxy):
    if form.action is None:
        return
    fq_attr_name = event_source_str("{}/{}".format(form.dev, form.attr))
    # change or periodic event?
    change_evt = form.evt_type == EventType.CHANGE_EVENT
    periodic_evt = form.evt_type == EventType.PERIODIC_EVENT
    # auto_conf requested?
    auto_conf = (change_evt or periodic_evt) and form.action.type == TangoEventSubscriptionAction.AUTO_CONFIGURE
    # attribute already polled?
    attr_is_polled = proxy.is_attribute_polled(form.attr)
    if auto_conf and not attr_is_polled:
        # print("no polling on device, auto-configuring event...")
        min_pp = TangoEventSubscriptionAction.MIN_POLLING_THRESHOLD_IN_MS
        if form.action.polling_period_ms < min_pp:
            txt = "Tango event auto-configuration error.\n"
            txt += "For safety reason, the auto-configured polling period must >= {:d} ".format(int(min_pp))
            txt += "[the specified value was {:d} ms]".format(int(form.action.polling_period_ms))
            raise Exception(txt)
    # get attr config.
    attr_cfg = proxy.get_attribute_config(form.attr)
    # case: periodic event
    if periodic_evt:
        # print("auto-configuring a PERIODIC event")
        if form.action.event_period_ms < form.action.polling_period_ms:
            txt = "Tango event auto-configuration error.\n"
            txt += "The event period must be greater or equal to the polling period"
            raise Exception(txt)
        attr_cfg.events.per_event.period = str(form.action.event_period_ms)
        # print("changing polling period on server side to {}".format(attr_cfg.events.per_event.period))
    # case: change event
    elif change_evt:
        if form.action.absolute_change == 0. and form.action.relative_change == 0.:
            txt = "Tango event auto-configuration error.\n"
            txt += "The both the absolute and relative change value are null"
            raise Exception(txt)
        # print("changing 'change event' thresholds for {}".format(attr_cfg.events.per_event.period))
        attr_cfg.events.ch_event.abs_change = str(form.action.absolute_change)
        attr_cfg.events.ch_event.rel_change = str(form.action.relative_change)
    if change_evt or periodic_evt:
        proxy.set_attribute_config(attr_cfg)
    # print("starting polling on server side for {}".format(fq_attr_name))
    proxy.poll_attribute(form.attr, int(form.action.polling_period_ms))
    # print("evt. successfully auto configured for {}".format(fq_attr_name))


class EventManager(object):

    logger = logging.getLogger('fs.utils.tango.EventManager')
    logger.setLevel(logging.ERROR)

    def __init__(self):
        self._subscriptions = {}  # {(<attribute_name>, <event_type>): TangoEventInfo}

    def event_from_data(self, data):
        evt_type = event_data_to_event_type(data)
        return TangoEvent(data=data, info=self.info(data.attr_name.lower(), evt_type))

    def subscribe(self, form, callback):
        """
        Tango event subscription.
        The 'form' argument is an instance of TangoEventSubscriptionForm
        Throws in case of error
        """
        dev_proxy = DeviceProxy(form.dev)
        fq_attr_name = event_source_str("{}/{}".format(dev_proxy.dev_name(), form.attr))
        if (fq_attr_name, form.evt_type) in self._subscriptions:
            try:
                self.unsubscribe(fq_attr_name, form.evt_type)
            except Exception as e:
                self.logger.error(e)
                raise
        tango_evt = TangoEventInfo(
            attr=form.attr,
            dev_proxy=dev_proxy,
            evt_type=form.evt_type,
            message_id=form.message_id,
            user_data=form.user_data,
        )
        pre_subscription_action(form, dev_proxy)
        self._subscriptions[(fq_attr_name, form.evt_type)] = tango_evt
        try:
            tango_evt.subscribe(callback)
        except:
            del self._subscriptions[(fq_attr_name, form.evt_type)]
            raise
        # print('successfully subscribed to {} for {}'.format(form.evt_type, fq_attr_name))

    def unsubscribe(self, fq_attr_name, evt_type):
        info = self._subscriptions.pop((fq_attr_name, evt_type), None)
        if info is not None:
            self.logger.debug('unsubscribing to {} for {}...'.format(evt_type, fq_attr_name))
            info.unsubscribe()
            self.logger.info('successfully unsubscribed to {} for {}'.format(evt_type, fq_attr_name))

    def unsubscribe_all(self):
        while self._subscriptions:
            (fq_attr_name, evt_type), info = self._subscriptions.popitem()
            self.logger.debug('unsubscribing to {} for {}...'.format(evt_type, fq_attr_name))
            info.unsubscribe()
            self.logger.info('successfully unsubscribed to %s for %s...', evt_type, fq_attr_name)

    def info(self, evt_source, evt_type):
        try:
            return self._subscriptions[(evt_source, evt_type)]
        except KeyError:
            raise LookupError("no subscription found for {!r} of type {!r}".format(evt_source, evt_type))


class TangoEventsConsumer(object):

    def __init__(self):
        self.event_manager = EventManager()

    def handle_message(self, msg):
        raise NotImplementedError

    def subscribe_event(self, form):
        try:
            self.event_manager.subscribe(form, self.__push_event)
        except Exception as err:
            # print("TangoEventsConsumer.subscribe_event failed!")
            # print(err)
            raise

    def unsubscribe_event(self, fq_attr_name, evt_type):
        self.event_manager.unsubscribe(fq_attr_name, evt_type)

    def unsubscribe_events(self):
        self.event_manager.unsubscribe_all()

    def __push_event(self, data):
        try:
            if data.attr_name.lower() != 'unknown':  # timeout
                event = self.event_manager.event_from_data(data)
                self._handle_event(event)
        except Exception as err:
            # print("unable to process the following event ", data, ", got the following error: ", err)
            pass

    def _handle_event(self, event):
        raise NotImplementedError


class TangoEventsConsumerTask(Task, TangoEventsConsumer):

    def __init__(self, *args, **kwargs):
        Task.__init__(self, *args, **kwargs)
        TangoEventsConsumer.__init__(self)

    def _handle_event(self, event):
        self.post(event.info.message_id, event)

    def handle_message(self, msg):
        raise NotImplementedError