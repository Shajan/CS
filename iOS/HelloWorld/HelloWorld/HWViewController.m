//
//  HWViewController.m
//  HelloWorld
//
//  Created by Shajan Dasan on 1/22/14.
//  Copyright (c) 2014 Shajan Dasan. All rights reserved.
//

#import "HWViewController.h"

@interface HWViewController ()

@end

@implementation HWViewController

- (void)viewDidLoad
{
    [super viewDidLoad];
	// Do any additional setup after loading the view, typically from a nib.
}

- (void)didReceiveMemoryWarning
{
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

- (IBAction)button:(id)sender
{
    _label.text = _text.text;
}
@end
