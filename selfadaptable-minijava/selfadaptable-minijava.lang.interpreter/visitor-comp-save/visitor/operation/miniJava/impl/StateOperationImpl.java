package visitor.operation.miniJava.impl;

import java.lang.String;
import miniJava.visitor.miniJava.Call;
import miniJava.visitor.miniJava.Context;
import miniJava.visitor.miniJava.Frame;
import miniJava.visitor.miniJava.MiniJavaFactory;
import miniJava.visitor.miniJava.ObjectInstance;
import miniJava.visitor.miniJava.State;
import org.eclipse.emf.ecoretools.ale.compiler.lib.EqualService;
import org.eclipse.emf.ecoretools.ale.compiler.lib.LogService;
import visitor.VisitorInterface;
import visitor.operation.miniJava.FrameOperation;
import visitor.operation.miniJava.StateOperation;

public class StateOperationImpl implements StateOperation {
	private final State it;

	private final VisitorInterface vis;

	public StateOperationImpl(State it, VisitorInterface vis) {
		this.it = it;
		this.vis = vis;
	}

	public Frame findCurrentFrame() {
		Frame result;
		if(EqualService.equals((this.it.getFrameCache()), (null))) {
			this.it.setFrameCache(((FrameOperation)this.it.getRootFrame().accept(vis)).findCurrentFrame());
		}
		result = ((Frame) (this.it.getFrameCache()));
		return result;
	}

	public Context findCurrentContext() {
		Context result;
		if(EqualService.equals((this.it.getContextCache()), (null))) {
			this.it.setContextCache(((FrameOperation)this.it.getRootFrame().accept(vis)).findCurrentContext());
		}
		result = ((Context) (this.it.getContextCache()));
		return result;
	}

	public void pushNewContext() {
		Context newContext = ((Context) (MiniJavaFactory.eINSTANCE.createContext()));
		Context currCtx = ((Context) (((StateOperation)this.it.accept(vis)).findCurrentContext()));
		if(!EqualService.equals((currCtx), (null))) {
			currCtx.setChildContext(newContext);
		}
		else {
			Frame cf = ((Frame) (((StateOperation)this.it.accept(vis)).findCurrentFrame()));
			cf.setRootContext(newContext);
		}
		this.it.setContextCache(newContext);
	}

	public void popCurrentContext() {
		Context currContext = ((Context) (((StateOperation)this.it.accept(vis)).findCurrentContext()));
		Context newCurrent = ((Context) (currContext.getParentContext()));
		currContext.setParentContext(null);
		this.it.setContextCache(newCurrent);
	}

	public void println(String str) {
		LogService.log(str);
		this.it.getOutputStream().getStream().add(str);
	}

	public void pushNewFrame(ObjectInstance receiver, Call c, Context newContext) {
		Frame newFrame = ((Frame) (MiniJavaFactory.eINSTANCE.createFrame()));
		newFrame.setInstance(receiver);
		newFrame.setCall(c);
		newFrame.setRootContext(newContext);
		((StateOperation)this.it.accept(vis)).findCurrentFrame().setChildFrame(newFrame);
		this.it.setFrameCache(newFrame);
		this.it.setContextCache(null);
	}

	public void popCurrentFrame() {
		Frame newCurrent = ((Frame) (((StateOperation)this.it.accept(vis)).findCurrentFrame().getParentFrame()));
		((StateOperation)this.it.accept(vis)).findCurrentFrame().setParentFrame(null);
		this.it.setContextCache(null);
		this.it.setFrameCache(newCurrent);
	}
}
