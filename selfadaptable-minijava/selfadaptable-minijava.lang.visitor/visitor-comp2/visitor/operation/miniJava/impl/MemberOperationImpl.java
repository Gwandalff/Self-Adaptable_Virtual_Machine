package visitor.operation.miniJava.impl;

import miniJava.visitor.miniJava.Member;
import visitor.VisitorInterface;
import visitor.operation.miniJava.MemberOperation;

public class MemberOperationImpl extends TypedDeclarationOperationImpl implements MemberOperation {
	private final Member it;

	private final VisitorInterface vis;

	public MemberOperationImpl(Member it, VisitorInterface vis) {
		super(it, vis);
		this.it = it;
		this.vis = vis;
	}
}
