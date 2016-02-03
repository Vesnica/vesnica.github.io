---
layout: post
title:  "AWS EC2 price cheating"
date:   2016-02-06 11:08:08 +0800
categories: aws ec2
permalink: aws_price_cheating
---

Recently I need some gpu power to support my machine learning research. After some investigation I found Amazon aws ec2 service offer gpu instances with a very attractive price. but the fact is not as seen.

In "Asia Pacific (Tokyo)" region, the spot price of g2.8xlarge instance(have 4 k520 gpu) is lower than 0.5$/hour at most of the times(0.188$ recently), but the price will boost 6~7 times(or even 20 times!) immediately if you make a spot request, and your request will failed miseably with "price is too low".

At first I thought it's a coincidence, "maybe some other ppl send request at the same time, so the price bar is raised". But after some experiments I found there is no other ppl, it's ME, the only requester, cause those beautiful spikes:

![Spikes](https://cloud.githubusercontent.com/assets/1401615/12772279/daa6b746-ca6a-11e5-8112-1538fb760ee2.PNG)

A price system is cheating (or broken) if a single user can cause such sharp fluctuations. Users wants to know the REAL current price all the time, but what they see now is just an honey illusion.
