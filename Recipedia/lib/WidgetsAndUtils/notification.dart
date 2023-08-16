import 'package:awesome_notifications/awesome_notifications.dart';
import 'package:flutter/material.dart';

class NotificationService {
  void initAwesomeNotification() async {
    AwesomeNotifications().initialize(
      null,
      [
        NotificationChannel(
          channelKey: 'key1',
          channelName: 'Recipedia',
          channelDescription: 'Testing Notification',
          defaultColor: const Color(0XFFff735c),
          ledColor: const Color(0XFFff735c),
          //soundSource: 'assets/notification.mp3',
          enableLights: true,
          enableVibration: true,
          playSound: true,
        )
      ],
    );
  }

  void requestPermission() async {
    AwesomeNotifications().isNotificationAllowed().then((allowed) {
      if (!allowed) {
        AwesomeNotifications().requestPermissionToSendNotifications();
      }
    });
  }

  void pushNotification(String message, String image) async {
    AwesomeNotifications().createNotification(
        content: NotificationContent(
      id: 1,
      channelKey: 'key1',
      title: 'Recipedia',
      body: message,
      bigPicture: image,
      notificationLayout: NotificationLayout.BigPicture,
    ));
  }
}
